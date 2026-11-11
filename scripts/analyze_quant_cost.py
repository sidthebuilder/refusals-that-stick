"""How much of a repair's safety benefit does quantization take away?

The retention ratio in ``analyze_grid.py`` answers this as a fraction, but its
denominator is the repair's BF16 benefit — tens of prompts — so the ratio is a
noisy point estimate and the sprint log declines to quote it as a finding. This
script reports the same quantity on the scale it is measured on, with an
interval, by estimating it **per prompt**::

    v_i = (repair_BF16 - drift_BF16)_i - (repair_quant - drift_quant)_i

The mean of ``v`` is the benefit lost to quantization, in refusal-rate points.
Because all four cells score the same prompts, resampling prompts (percentile
bootstrap, seeded) gives an interval that respects the pairing.

Comparing the repair against the drifted model *at the same precision* on both
sides is what isolates the repair: quantization's effect on the model generally
cancels.

Usage:
    python scripts/analyze_quant_cost.py _scratch/cells_test/test --split test \\
        --precisions Q4_K_M --out quant_cost_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.analysis import quantization_cost  # noqa: E402

PRECISIONS = ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells_root", type=Path)
    ap.add_argument("--drift", default="drift-dose3")
    ap.add_argument("--sparse", default="safemerge-dose3-s1")
    ap.add_argument("--dense", default="resta-dose3-s0.25")
    ap.add_argument("--split", default="test")
    ap.add_argument("--precisions", default=",".join(PRECISIONS))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="quant_cost_test.json")
    args = ap.parse_args()

    root, split = args.cells_root, args.split
    arms = [("sparse", args.sparse), ("dense", args.dense)]
    wanted = [p for p in args.precisions.split(",") if p]

    def has(model: str, precision: str) -> bool:
        return (root / model / precision / f"judge_redteam_hi_{split}.jsonl").exists()

    cells = [(arm, name, p) for arm, name in arms for p in wanted
             if has(name, p) and has(args.drift, p)
             and has(name, "BF16") and has(args.drift, "BF16")]
    if not cells:
        print(f"no cells found under {root}")
        return 1
    # One interval per (arm, precision, language): widen each to hold the
    # family-wise error at 5% across the set actually computed.
    k = len(cells) * 2
    fw_level = 1 - 0.05 / k

    results = []
    for arm, name, p in cells:
        for lang in ("hi", "en"):
            common = dict(drift_bf16=root / args.drift / "BF16",
                          repair_bf16=root / name / "BF16",
                          drift_quant=root / args.drift / p,
                          repair_quant=root / name / p,
                          task=f"redteam_{lang}_{split}", iters=args.iters, seed=args.seed)
            cost = quantization_cost(**common)
            wide = quantization_cost(**common, level=fw_level)
            # Standard error implied by the bootstrap interval, and the sample size
            # at which an attenuation of the size observed would be resolvable
            # (two-sided 5%, 80% power: |effect| >= 2.8 * SE). This is what makes
            # a null here interpretable: it says whether the experiment could have
            # seen the effect, not merely that it did not.
            se = (cost["ci95"][1] - cost["ci95"][0]) / (2 * 1.959963984540054)
            need = (None if abs(cost["benefit_lost"]) < 1e-12 or se == 0 else
                    round(cost["n"] * (se / (abs(cost["benefit_lost"]) / 2.8)) ** 2))
            cost.update({"arm": arm, "repair": name, "precision": p, "lang": lang,
                         "ci_familywise": wide["ci95"],
                         "significant_familywise": wide["significant"],
                         "familywise_level": fw_level,
                         "se": se, "n_for_80pct_power": need})
            results.append(cost)

    out = {
        "split": split,
        "estimand": "benefit_lost = (repair-drift)@BF16 - (repair-drift)@precision, "
                    "per prompt, on valid refusal for red-team prompts",
        "why_not_the_ratio": "retention divides by the BF16 benefit, a count in the tens; "
                             "this is the same quantity on its measured scale, with an interval",
        "bootstrap": {"iters": args.iters, "seed": args.seed, "level": 0.95,
                      "familywise_level": fw_level, "n_intervals": k},
        "results": results,
    }
    (ROOT / "results" / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"benefit lost to quantization, {split} split, bootstrap {args.iters} draws "
          f"(seed {args.seed})")
    print(f"{'arm':7s} {'prec':7s} {'lang':4s} {'n':>5s} {'bf16':>8s} {'quant':>8s} "
          f"{'lost':>8s} {'95% CI':>18s} {'share':>7s} {'sig':>5s} {'n@80%':>8s}")
    for r in results:
        share = r["share_lost"] if r["share_lost"] is not None else float("nan")
        need = r["n_for_80pct_power"]
        print(f"{r['arm']:7s} {r['precision']:7s} {r['lang']:4s} {r['n']:5d} "
              f"{r['benefit_bf16']:+8.3f} {r['benefit_quant']:+8.3f} "
              f"{r['benefit_lost']:+8.3f} "
              f"[{r['ci95'][0]:+7.3f},{r['ci95'][1]:+7.3f}] {share:7.2f} "
              f"{str(r['significant']):>5s} {('-' if need is None else str(need)):>8s}")
    print("n@80% = prompts per language needed for an attenuation this size to be")
    print("resolvable at 80% power, from the bootstrap SE.")
    print(f"\nfamily-wise intervals use level {fw_level:.4f} (Bonferroni over {k} intervals)")
    print(f"wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
