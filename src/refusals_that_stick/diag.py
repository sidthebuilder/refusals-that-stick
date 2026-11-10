"""Decoding diagnostic: does parallel serving change what the model generates?

The first baseline cells produced outputs like ``"Che question asks about..."``
for an MMLU item whose answer is C. The continuation is coherent for a first word
of "The", but the first character is the answer letter. Greedy looping does not
do that. A sequence reading another sequence's KV-cache entries would. Cohere2
uses sliding-window attention on three of every four layers, and llama.cpp's
reduced-size SWA cache shared across parallel slots is a plausible source.

So the same prompts are decoded under several server configurations and
compared token by token against a single-slot reference, where no sharing is
possible:

* ``np1``          one slot, sequential requests: the reference;
* ``np32``         32 slots, the configuration the first cells used;
* ``np32_swafull`` 32 slots with ``--swa-full`` (full-size SWA cache);
* ``np32_nocache`` 32 slots, ``cache_prompt`` off (no prefix reuse across requests).

Token ids come back from ``/completion`` with ``return_tokens`` and are
rendered with ``/detokenize``, so hidden special tokens are visible.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .serve import LLAMA_BIN, LlamaServer, _session

# Fire ends a response with <|END_RESPONSE|> (id 261001), sometimes followed by
# <EOS_TOKEN> (id 3); llama.cpp's end-of-generation set for the GGUF holds only
# <|END_OF_TURN_TOKEN|> (id 6). The override registers the other two as EOT and
# EOM, which llama.cpp also treats as end of generation. The GGUF file itself
# is not modified.
EOG_FIX = ["--override-kv", "tokenizer.ggml.eot_token_id=int:261001",
           "--override-kv", "tokenizer.ggml.eom_token_id=int:3"]

CONFIGS: dict[str, dict[str, Any]] = {
    "np1": {"n_parallel": 1, "extra": [], "cache_prompt": True},
    "np32": {"n_parallel": 32, "extra": [], "cache_prompt": True},
    "np1_eogfix": {"n_parallel": 1, "extra": EOG_FIX, "cache_prompt": True},
    "np32_eogfix": {"n_parallel": 32, "extra": EOG_FIX, "cache_prompt": True},
}


def eog_metadata(gguf: str, hf_dir: str | None) -> dict[str, Any]:
    """Special-token ids in the GGUF, their texts, and the Hugging Face stop set."""
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(gguf, "r")

    def field(key: str):
        f = reader.fields.get(key)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:  # noqa: BLE001
            return "unreadable"

    tokens = field("tokenizer.ggml.tokens") or []
    keys = ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id",
            "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token"]
    out: dict[str, Any] = {k: field(k) for k in keys}
    out["texts"] = {str(i): (tokens[i] if i < len(tokens) else None) for i in (3, 6, 261001)}
    if hf_dir:
        for name in ("generation_config.json", "config.json"):
            p = Path(hf_dir) / name
            if p.exists():
                cfg = json.loads(p.read_text())
                out[f"hf_{name}"] = {k: cfg.get(k) for k in
                                     ("eos_token_id", "bos_token_id", "pad_token_id")}
    return out


def server_help() -> str:
    r = subprocess.run([str(LLAMA_BIN / "llama-server"), "--help"], capture_output=True,
                       text=True, timeout=120)
    text = (r.stdout or "") + (r.stderr or "")
    keep = [ln for ln in text.splitlines()
            if any(k in ln for k in ("swa", "kv-unified", "kvu", "cache-reuse", "parallel",
                                     "cache-prompt", "repeat-penalty", "special"))]
    return "\n".join(keep)


def _complete(srv: LlamaServer, prompt: str, n_predict: int, cache_prompt: bool) -> dict:
    data = srv._post("/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 0,
        "cache_prompt": cache_prompt, "return_tokens": True, "stream": False,
    })
    return {"text": data.get("content") or "", "tokens": data.get("tokens") or [],
            "stop_type": data.get("stop_type")}


def _pieces(srv: LlamaServer, tokens: list[int]) -> list[str]:
    if not tokens:
        return []
    r = _session().post(f"{srv.base}/detokenize",
                        json={"tokens": tokens, "with_pieces": True}, timeout=60)
    r.raise_for_status()
    data = r.json()
    pieces = data.get("pieces")
    if isinstance(pieces, list):
        return [p.get("piece") if isinstance(p, dict) else str(p) for p in pieces]
    return [data.get("content", "")]


def run(gguf: str, prompts: list[dict], out_dir: str, hf_dir: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"eog_metadata": eog_metadata(gguf, hf_dir), "configs": {}}

    # Render every prompt once, through the GGUF's own template, so all
    # configurations decode byte-identical inputs.
    with LlamaServer(gguf, n_parallel=1) as srv:
        rendered = [srv.apply_template([{"role": "user", "content": p["prompt"]}])
                    for p in prompts]

    results: dict[str, list[dict]] = {}
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        try:
            with LlamaServer(gguf, n_parallel=cfg["n_parallel"],
                             extra_args=cfg["extra"]) as srv:
                def one(i: int) -> dict:
                    r = _complete(srv, rendered[i], prompts[i]["n_predict"], cfg["cache_prompt"])
                    return {"id": prompts[i]["id"], **r}

                rows = srv.map(one, list(range(len(prompts))),
                               workers=max(1, cfg["n_parallel"]))
                for row in rows[:8]:  # show hidden special tokens for a sample
                    row["pieces"] = _pieces(srv, row["tokens"][:32])
            results[name] = rows
            report["configs"][name] = {"ok": True, "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001 - a config may be unsupported; record why
            report["configs"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500]}

    for name, rows in results.items():
        stops = [r.get("stop_type") for r in rows]
        report["configs"][name]["stop_types"] = {s: stops.count(s) for s in set(stops)}
        report["configs"][name]["mean_tokens"] = (
            sum(len(r["tokens"]) for r in rows) / max(1, len(rows)))
    # Each fixed configuration is compared with its own single-slot reference.
    for name, rows in results.items():
        base = "np1_eogfix" if name.endswith("_eogfix") else "np1"
        ref = {r["id"]: r for r in results.get(base, [])}
        if name == base or not ref:
            continue
        same, first_div = 0, []
        for r in rows:
            a, b = ref[r["id"]]["tokens"], r["tokens"]
            if a == b:
                same += 1
                continue
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            first_div.append(k)
        report["configs"][name].update(
            identical_to_np1=same, n=len(rows),
            first_divergence_tokens=sorted(first_div)[:40],
        )
    (out / "diag_outputs.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (out / "diag_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    report["samples"] = {name: [{"id": r["id"], "text": r["text"][:200],
                                 "pieces": r.get("pieces", [])[:16]} for r in rows[:6]]
                         for name, rows in results.items()}
    return report
