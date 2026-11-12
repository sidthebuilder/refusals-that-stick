"""No-harm check for the protected file (fix 1), on the dev split.

The edit-aware protected Q4_K_M file keeps the sparse repair's 15 edited tensors at
Q8_0. This compares its behaviour, prompt by prompt, with the plain Q4_K_M file of
the same repair and with the drifted model at Q4_K_M. It is a no-harm check, as
logged before the run: 150 prompts per language cannot resolve a safety effect of
the size at stake (see results/quant_cost_test.json), so only a large change in
either direction would show up.

    python scripts/analyze_protection_cell.py
Writes results/protection_behaviour.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.analysis import cell_metrics, compare  # noqa: E402

CELLS = ROOT / "_scratch" / "cells_v2" / "dev"
REPAIR, DRIFT = "safemerge-dose3-s1", "drift-dose3"
CELL = {"protected": CELLS / REPAIR / "Q4_K_M-protect-edit",
        "plain": CELLS / REPAIR / "Q4_K_M",
        "drift": CELLS / DRIFT / "Q4_K_M"}


def main() -> int:
    metrics = {k: cell_metrics(v, split="dev") for k, v in CELL.items()}
    tests = {}
    for lang in ("hi", "en"):
        task = f"redteam_{lang}_dev"
        tests[f"protected_vs_plain_{lang}"] = compare(CELL["plain"], CELL["protected"], task, "refused")
        tests[f"protected_vs_drift_{lang}"] = compare(CELL["drift"], CELL["protected"], task, "refused")
    out = {"split": "dev", "n_per_language": tests["protected_vs_plain_hi"]["n"],
           "metric": "valid refusal on red-team prompts; paired exact McNemar",
           "metrics": metrics, "tests": tests}
    (ROOT / "results" / "protection_behaviour.json").write_text(json.dumps(out, indent=2),
                                                                encoding="utf-8")
    keys = ("redteam_hi_refusal", "redteam_en_refusal", "xstest_hi_refusal", "xstest_en_refusal",
            "mmlu_hi", "mmlu_en", "ppl_hi")
    print(f"{'cell':10s} " + " ".join(f"{k:>18s}" for k in keys))
    for name, m in metrics.items():
        print(f"{name:10s} " + " ".join(f"{(m.get(k) if m.get(k) is not None else float('nan')):18.3f}"
                                        for k in keys))
    for name, t in tests.items():
        print(f"{name:24s} delta={t['delta']:+.3f} p={t['p_mcnemar']:.3f} n={t['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
