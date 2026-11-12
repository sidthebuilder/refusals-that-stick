"""Return on investment: serving the repaired small model vs paying a closed API.

Throughput is measured, not assumed: the test-split cells record every request's
prompt and completion token counts and the wall time of the generation phase on
one NVIDIA L40S (llama.cpp, 32 parallel slots, greedy decoding). Prices are list
prices checked on 12 Nov 2026; sources are in PRICES below.

The comparison uses the same token counts for every option. Our prompt count
includes Tiny Aya's own system preamble (about 360 tokens), which a closed API
would not need, so the API input cost is an upper bound.

Tiny Aya is CC-BY-NC: self-hosting it is for research and non-commercial use.

    python scripts/analyze_roi.py _scratch/cells_test/test
Writes results/roi.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# USD. Checked 12 Nov 2026.
GPU_PER_HOUR = {
    "Modal L40S": (1.95, "https://modal.com/blog/introducing-l40s ; "
                         "https://computeprices.com/providers/modal/gpus/l40s"),
}
API_PER_MTOK = {  # (input, output) per 1M tokens, standard tier, short context
    "gpt-6-luna": (0.10, 0.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-6.1-sol": (2.00, 10.00),
}
API_SOURCE = "https://developers.openai.com/api/docs/pricing"
REQUESTS = 10_000
MODELS = ("fire", "safemerge-dose3-s1", "resta-dose3-s0.25")


def measure(cells_root: Path, model: str, precision: str) -> dict:
    cells = {c["gguf"]: c for c in json.loads((ROOT / "results" / "cells_test_test.json")
                                              .read_text(encoding="utf-8"))}
    timing = cells[f"/vol/gguf/{model}/{precision}.gguf"]["timings"]
    n = ptok = ctok = 0
    for gen in sorted((cells_root / model / precision).glob("gen_*.jsonl")):
        for line in gen.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                n += 1
                ptok += g.get("prompt_tokens") or 0
                ctok += g.get("completion_tokens") or 0
    s = timing["generate_s"]
    return {"requests": n, "prompt_tokens": ptok, "completion_tokens": ctok,
            "generate_seconds": s, "requests_per_second": n / s,
            "completion_tokens_per_second": ctok / s,
            "mean_prompt_tokens": ptok / n, "mean_completion_tokens": ctok / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells_root", type=Path)
    args = ap.parse_args()
    if not args.cells_root.exists():
        print(f"no cells at {args.cells_root}", file=sys.stderr)
        return 1

    rows = []
    for model in MODELS:
        for precision in ("BF16", "Q4_K_M"):
            m = measure(args.cells_root, model, precision)
            gpu, (price, _) = next(iter(GPU_PER_HOUR.items()))
            hours = REQUESTS / m["requests_per_second"] / 3600
            m.update(model=model, precision=precision, gpu=gpu,
                     cost_per_10k=hours * price,
                     cost_per_mtok_output=price / (m["completion_tokens_per_second"] * 3600 / 1e6))
            rows.append(m)

    ref = next(r for r in rows if r["model"] == "fire" and r["precision"] == "Q4_K_M")
    pin = ref["mean_prompt_tokens"] * REQUESTS / 1e6
    pout = ref["mean_completion_tokens"] * REQUESTS / 1e6
    apis = {name: {"input_per_mtok": i, "output_per_mtok": o,
                   "cost_per_10k": pin * i + pout * o,
                   "cost_per_10k_output_only": pout * o}
            for name, (i, o) in API_PER_MTOK.items()}
    out = {"requests": REQUESTS, "token_mix_from": "fire Q4_K_M, test split",
           "mean_prompt_tokens": ref["mean_prompt_tokens"],
           "mean_completion_tokens": ref["mean_completion_tokens"],
           "self_hosted": rows, "apis": apis,
           "gpu_prices": {k: {"usd_per_hour": v[0], "source": v[1]} for k, v in GPU_PER_HOUR.items()},
           "api_source": API_SOURCE, "prices_checked": "2026-11-12",
           "licence_note": "Tiny Aya is CC-BY-NC: self-hosting is for research and non-commercial use"}
    (ROOT / "results" / "roi.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Token mix (fire Q4_K_M, test split): {ref['mean_prompt_tokens']:.0f} prompt, "
          f"{ref['mean_completion_tokens']:.0f} completion tokens per request\n")
    print(f"{'self-hosted on one L40S':34s} {'req/s':>6s} {'tok/s':>6s} {'$/10k req':>10s} {'$/1M out':>9s}")
    for r in rows:
        print(f"  {r['model']:22s} {r['precision']:7s} {r['requests_per_second']:6.2f} "
              f"{r['completion_tokens_per_second']:6.0f} {r['cost_per_10k']:10.2f} "
              f"{r['cost_per_mtok_output']:9.2f}")
    print(f"\n{'closed API, same tokens':34s} {'$/10k req':>10s} {'(output only)':>14s}")
    for name, a in apis.items():
        print(f"  {name:32s} {a['cost_per_10k']:10.2f} {a['cost_per_10k_output_only']:14.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
