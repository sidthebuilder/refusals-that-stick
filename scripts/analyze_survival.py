"""Weight-level survival: H0 (sparsity drives erasure) and H6 (weights predict behaviour).

Reads the pooled survival summaries (results/survival_<tag>.json) and the
behavioural grid (results/grid_analysis.json), then tests:

* **H0** — a sparse repair loses more per-weight permanence under 4-bit than a
  dense one. Here the sparse arm edits 8.2% of weights (SafeMerge, 15 tensors)
  and the dense arm edits every tensor (RESTA). Falsified if sparse permanence
  is greater than or equal to dense permanence at Q4_K_M.
* **H6** — per-weight permanence predicts behavioural retention across cells,
  Spearman rho > 0.5.

Both headline weight metrics are per-edited-weight and bounded in [0, 1]:
``permanence`` (the deployed value moved at all) and ``directional_permanence``
(it moved the intended way). See survival.py for the two metrics that were
tested and rejected on day zero.

The chance level for ``directional_permanence`` is ``permanence / 2``, not 0.5:
a weight that did not move counts as a failure, and a weight that moved at
random goes the intended way half the time. So the comparison with chance is
made on ``direction_among_moved`` = directional_permanence / permanence, whose
chance level is 0.5. Edited weights then split three ways: erased (did not
move), moved the intended way, moved against it.

Usage:
    python scripts/analyze_survival.py [--tag main]
Writes results/survival_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.stats import spearman  # noqa: E402

H0_PRECISION = "Q4_K_M"
H6_THRESHOLD = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    surv = json.loads((ROOT / "results" / f"survival_{args.tag}.json").read_text())
    grid_path = ROOT / "results" / "grid_analysis.json"
    grid = json.loads(grid_path.read_text()) if grid_path.exists() else {"survival": []}

    ok = [s for s in surv if "error" not in s and s.get("permanence") is not None]
    failed = [s for s in surv if "error" in s]

    # Arm labels come from how many weights the repair actually edits, measured
    # from the files rather than assumed from the method name.
    arms = {}
    for s in ok:
        share = s["share_of_weights_edited"]
        arms[s["repair"]] = "sparse" if share < 0.5 else "dense"

    rows = []
    for s in ok:
        perm, dirp = s["permanence"], s["directional_permanence"]
        rows.append({
            "repair": s["repair"], "arm": arms[s["repair"]], "precision": s["qtype"],
            "share_of_weights_edited": s["share_of_weights_edited"],
            "n_tensors_edited": s["n_tensors_edited"],
            "permanence": perm,
            "directional_permanence": dirp,
            "erased_fraction": s["erased_fraction"],
            # Of the edited weights that moved, the share that moved the intended
            # way. Chance is 0.5 here; for directional_permanence it is perm / 2.
            "direction_among_moved": dirp / perm if perm > 0 else None,
            "chance_directional_permanence": perm / 2,
            "moved_against_fraction": perm - dirp,
        })
    rows.sort(key=lambda r: (r["arm"], r["precision"]))

    # --- H0: sparse vs dense permanence at the headline precision ---------------
    h0 = {"precision": H0_PRECISION, "threshold_note":
          "falsified if sparse >= dense at this precision"}
    for arm in ("sparse", "dense"):
        r = next((x for x in rows if x["arm"] == arm and x["precision"] == H0_PRECISION), None)
        if r:
            h0[arm] = {"repair": r["repair"], "share_edited": r["share_of_weights_edited"],
                       "permanence": r["permanence"],
                       "directional_permanence": r["directional_permanence"]}
    if "sparse" in h0 and "dense" in h0:
        h0["sparse_lower_permanence"] = h0["sparse"]["permanence"] < h0["dense"]["permanence"]
        h0["sparse_lower_directional"] = (h0["sparse"]["directional_permanence"]
                                          < h0["dense"]["directional_permanence"])
        h0["supported"] = h0["sparse_lower_directional"]
    # Direction across every precision, not just the headline one.
    h0["by_precision"] = []
    for p in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"):
        s = next((x for x in rows if x["arm"] == "sparse" and x["precision"] == p), None)
        d = next((x for x in rows if x["arm"] == "dense" and x["precision"] == p), None)
        if s and d:
            h0["by_precision"].append({
                "precision": p,
                "sparse_directional": s["directional_permanence"],
                "dense_directional": d["directional_permanence"],
                "sparse_lower": s["directional_permanence"] < d["directional_permanence"],
                "sparse_direction_among_moved": s["direction_among_moved"],
                "dense_direction_among_moved": d["direction_among_moved"],
                "sparse_lower_among_moved": (s["direction_among_moved"]
                                             < d["direction_among_moved"])})
    h0["n_precisions_sparse_lower"] = sum(1 for x in h0["by_precision"] if x["sparse_lower"])
    h0["n_precisions_sparse_lower_among_moved"] = sum(
        1 for x in h0["by_precision"] if x["sparse_lower_among_moved"])
    h0["n_precisions"] = len(h0["by_precision"])

    # --- H6: do the weight metrics predict the behaviour? ----------------------
    ret = {(e["repair"], e["precision"]): e for e in grid.get("survival", [])}
    paired = []
    for r in rows:
        e = ret.get((r["repair"], r["precision"]))
        if not e:
            continue
        for lang in ("hi", "en"):
            if e.get(f"retention_{lang}") is not None:
                paired.append({**{k: r[k] for k in ("repair", "arm", "precision", "permanence",
                                                    "directional_permanence")},
                               "lang": lang, "retention": e[f"retention_{lang}"]})
    h6 = {"threshold": H6_THRESHOLD, "n_pairs": len(paired)}
    for metric in ("permanence", "directional_permanence"):
        for lang in (None, "hi", "en"):
            sel = [p for p in paired if lang is None or p["lang"] == lang]
            rho = spearman([p[metric] for p in sel], [p["retention"] for p in sel])
            h6[f"rho_{metric}" + (f"_{lang}" if lang else "_all")] = rho
    rho_all = h6.get("rho_directional_permanence_all")
    h6["supported"] = rho_all is not None and rho_all > H6_THRESHOLD
    h6["pairs"] = paired

    # --- Per-module breakdown at the headline precision ------------------------
    modules = {}
    for s in ok:
        if s["qtype"] != H0_PRECISION:
            continue
        modules[s["repair"]] = {
            m: {"n_tensors": v["n_tensors"], "n_edited": v["n_edited"],
                "directional_permanence": v["directional_permanence"]}
            for m, v in sorted((s.get("by_module") or {}).items(),
                               key=lambda kv: -(kv[1]["n_edited"] or 0))[:10]
        }

    out = {"tag": args.tag, "failed_jobs": failed, "rows": rows,
           "H0_sparsity_drives_erasure": h0, "H6_weights_predict_behaviour": h6,
           "modules_at_headline_precision": modules}
    (ROOT / "results" / "survival_analysis.json").write_text(json.dumps(out, indent=2))

    if failed:
        print(f"FAILED {len(failed)} jobs: {[(f['repair'], f['qtype']) for f in failed]}\n")
    print(f"{'arm':8s} {'repair':24s} {'prec':7s} {'edited':>7s} {'perm':>6s} "
          f"{'dirperm':>8s} {'chance':>7s} {'dir|moved':>9s} {'erased':>7s} {'against':>8s}")
    for r in rows:
        print(f"{r['arm']:8s} {r['repair']:24s} {r['precision']:7s} "
              f"{r['share_of_weights_edited']:7.3f} {r['permanence']:6.3f} "
              f"{r['directional_permanence']:8.3f} {r['chance_directional_permanence']:7.3f} "
              f"{r['direction_among_moved']:9.3f} {r['erased_fraction']:7.3f} "
              f"{r['moved_against_fraction']:8.3f}")
    print("\nH0:", json.dumps({k: v for k, v in h0.items() if k != "by_precision"}, indent=2))
    print("H0 by precision:", json.dumps(h0["by_precision"]))
    print("H6:", json.dumps({k: v for k, v in h6.items() if k != "pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
