"""Convert labels/labels_table.txt into labels/labels.json, the format score_labels.py reads.

Refuses to write anything if the table is out of step with the sheet: every row
number must exist in labels/label_items.json, and every row's kind (redteam or
xstest) must match the kind recorded for that item. One skipped or misnumbered
row would otherwise shift every label after it onto the wrong response.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels"


def main() -> int:
    items = {it["item"]: it for it in
             json.loads((LABELS / "label_items.json").read_text(encoding="utf-8"))}
    rows = [ln.split() for ln in (LABELS / "labels_table.txt").read_text(encoding="utf-8")
            .splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    out: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for parts in rows:
        if len(parts) != 4:
            problems.append(f"malformed row: {parts}")
            continue
        num, kind, harmful, refused = parts
        key = f"item{int(num):03d}"
        it = items.get(key)
        if it is None:
            problems.append(f"{key}: not in label_items.json")
            continue
        if it["kind"] != kind:
            problems.append(f"{key}: table says {kind}, sheet says {it['kind']}")
            continue
        if refused not in ("yes", "no"):
            problems.append(f"{key}: refused must be yes/no, got {refused!r}")
            continue
        lab = {"refused": refused}
        if kind == "redteam":
            if harmful not in ("yes", "no"):
                problems.append(f"{key}: redteam needs harmful yes/no, got {harmful!r}")
                continue
            lab["harmful"] = harmful
        elif harmful != "-":
            problems.append(f"{key}: xstest items have no harmful question, got {harmful!r}")
            continue
        if key in out:
            problems.append(f"{key}: duplicate row")
            continue
        out[key] = lab
    missing = sorted(set(items) - set(out))
    if missing:
        problems.append(f"{len(missing)} items unlabelled, e.g. {missing[:5]}")
    if problems:
        print("NOT WRITTEN. Problems:")
        for p in problems:
            print("  ", p)
        return 1
    (LABELS / "labels.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_rt = sum(1 for v in out.values() if "harmful" in v)
    print(f"wrote labels/labels.json: {len(out)} items ({n_rt} redteam, {len(out) - n_rt} xstest), "
          "all kinds match the sheet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
