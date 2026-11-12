"""Fix 1 (H4): keep chosen tensors at Q8_0 inside a Q4_K_M or Q4_0 file.

Computed exactly from the per-tensor survival maps (see protection.py), so it needs
no GPU. Arms, fixed in SPRINT_LOG.md before any number was computed:

1. edit-aware:    the tensors the repair edits
2. circuit-aware: CircuitKIT's refusal circuit (fix 1 as pre-registered)
3. random:        random quantized tensors with the same total parameters as arm 1
                  (equal bytes at Q8_0); seeds 0-2 as registered, plus 10,000 seeds

    python scripts/analyze_protection.py [--maps results/survival_maps]
Writes results/protection_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.protection import (circuit_tensor_names, pooled_survival,  # noqa: E402
                                            protected_survival, random_matched_set,
                                            size_bytes)

REPAIRS = {"sparse": "safemerge-dose3-s1", "dense": "resta-dose3-s0.25"}
BASES = ("Q4_K_M", "Q4_0")
REGISTERED_SEEDS = (0, 1, 2)
N_SEEDS = 10000


def rows_of(maps: Path, repair: str, precision: str) -> list[dict]:
    return json.loads((maps / repair / f"{precision}.json").read_text(encoding="utf-8"))["tensors"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, default=ROOT / "results" / "survival_maps")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    circ = json.loads((ROOT / "results" / "circuit_report.json").read_text(
        encoding="utf-8"))["selection"]["chosen"]
    circuit = circuit_tensor_names(circ["layer_subset"], circ["target_modules"])
    sparse_edit = sorted(r["name"] for r in rows_of(args.maps, REPAIRS["sparse"], "Q4_K_M")
                         if r.get("n_edited", 0) > 0)

    out: dict = {"assumption": "--tensor-type leaves every other tensor's type unchanged; "
                               "checked by building one protected file",
                 "sparse_edited_tensors": sparse_edit, "circuit_tensors": circuit,
                 "circuit_layers": circ["layer_subset"], "results": {}}
    for base in BASES:
        # RESTA edits every tensor, so its maps carry the GGML type of all 290.
        types = rows_of(args.maps, REPAIRS["dense"], base)
        by_name = {r["name"]: r for r in types}
        params = lambda names: sum(by_name[n]["n_params"] for n in names)  # noqa: E731
        q8_full = size_bytes(types, [r["name"] for r in types])
        base_size = size_bytes(types)
        target = params(sparse_edit)
        seeds = {s: random_matched_set(types, target, s) for s in range(args.seeds)}
        arms = {"none": [], "edit_aware": sparse_edit, "circuit": circuit}
        res: dict = {"base_payload_bytes": base_size, "q8_payload_bytes": q8_full,
                     "target_params": target, "arms": {}}
        for arm_label, repair in REPAIRS.items():
            base_rows = rows_of(args.maps, repair, base)
            q8_rows = rows_of(args.maps, repair, "Q8_0")
            block: dict = {"full_q8_0": pooled_survival(q8_rows)}
            for name, protect in arms.items():
                s = protected_survival(base_rows, q8_rows, protect)
                s.update(params_protected=params(protect),
                         extra_bytes=size_bytes(types, protect) - base_size,
                         size_increase=(size_bytes(types, protect) - base_size) / base_size)
                block[name] = s
            dist, moved, n_tensors, sizes = [], [], [], []
            for seed, protect in seeds.items():
                s = protected_survival(base_rows, q8_rows, protect)
                dist.append(s["directional_permanence"])
                moved.append(s["direction_among_moved"])
                n_tensors.append(len(protect))
                sizes.append((size_bytes(types, protect) - base_size) / base_size)
                if seed in REGISTERED_SEEDS:
                    s.update(params_protected=params(protect), size_increase=sizes[-1])
                    block[f"random_seed{seed}"] = s

            def summary(xs: list[float]) -> dict:
                xs = sorted(xs)
                return {"mean": sum(xs) / len(xs), "p05": xs[int(0.05 * len(xs))],
                        "p95": xs[int(0.95 * len(xs)) - 1]}

            c = block["circuit"]["directional_permanence"]
            block["random_distribution"] = {
                "n": len(dist), **summary(dist),
                "direction_among_moved": summary(moved),
                "n_tensors": summary(n_tensors),
                "size_increase": summary(sizes),
                "share_at_or_above_circuit": sum(1 for d in dist if d >= c) / len(dist)}
            block["H4_circuit_beats_registered_seeds"] = all(
                c > block[f"random_seed{s}"]["directional_permanence"] for s in REGISTERED_SEEDS)
            res["arms"][arm_label] = block
        out["results"][base] = res

    out["H4_supported"] = all(out["results"][b]["arms"]["sparse"]["H4_circuit_beats_registered_seeds"]
                              for b in BASES)
    (ROOT / "results" / "protection_analysis.json").write_text(json.dumps(out, indent=2),
                                                               encoding="utf-8")

    print(f"sparse repair edits {len(sparse_edit)} tensors; circuit has {len(circuit)} tensors, "
          f"{len(set(circuit) & set(sparse_edit))} shared")
    for base in BASES:
        res = out["results"][base]
        print(f"\n=== base {base}: payload {res['base_payload_bytes'] / 1e9:.3f} GB, "
              f"full Q8_0 {res['q8_payload_bytes'] / 1e9:.3f} GB "
              f"(+{res['q8_payload_bytes'] / res['base_payload_bytes'] - 1:.1%}) ===")
        for arm_label, block in res["arms"].items():
            print(f"  [{arm_label}]  {'arm':16s} {'dirperm':>8s} {'dir|moved':>9s} "
                  f"{'erased':>7s} {'edited prot':>11s} {'size':>7s}")
            for name in ("none", "edit_aware", "circuit", "random_seed0", "random_seed1",
                         "random_seed2", "full_q8_0"):
                s = block[name]
                size = s.get("size_increase")
                print(f"  {'':9s} {name:16s} {s['directional_permanence']:8.3f} "
                      f"{s['direction_among_moved']:9.3f} {s['erased_fraction']:7.3f} "
                      f"{str(s.get('n_edited_tensors_protected', '-')):>11s} "
                      f"{('+%.1f%%' % (100 * size)) if size is not None else '-':>7s}")
            d = block["random_distribution"]
            am = d["direction_among_moved"]
            print(f"  {'':9s} random x{d['n']}: dirperm mean {d['mean']:.3f} "
                  f"[{d['p05']:.3f}, {d['p95']:.3f}]; dir|moved mean {am['mean']:.3f} "
                  f"[{am['p05']:.3f}, {am['p95']:.3f}]; tensors mean {d['n_tensors']['mean']:.1f}; "
                  f"size mean +{100 * d['size_increase']['mean']:.1f}%; share >= circuit "
                  f"{d['share_at_or_above_circuit']:.3f}; circuit beats seeds 0-2: "
                  f"{block['H4_circuit_beats_registered_seeds']}")
    print(f"\nH4 supported (sparse, both bases): {out['H4_supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
