"""Staging of the published results: per-prompt labels with no response text.

PREREGISTRATION.md §7 allows the release of labels and metadata but not model
generations: the drifted model complies with harmful requests, so its text is
withheld while the numbers behind every statistic are published.

Keeping the staging logic here, rather than in the publishing script, means the
rule can be tested. The field allow-list is the mechanism (only named fields are
copied) and :func:`build_results_release` additionally refuses to write if a
response-bearing field appears anyway, so a future field rename cannot leak text
by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["GEN_KEEP", "JUDGE_KEEP", "BANNED", "collect_records", "build_results_release"]

# Safe to publish from a generation record: everything except the prompt and the
# response text.
GEN_KEEP = ("id", "item", "lang", "split", "task", "finish_reason", "prompt_tokens",
            "completion_tokens", "degenerate", "devanagari_ratio", "answer", "choice",
            "correct")
JUDGE_KEEP = ("id", "safety", "refusal", "categories", "parsed")
# Fields that carry, or could carry, prompt or completion text.
BANNED = frozenset({"prompt", "response", "raw", "backtranslation"})


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def collect_records(cells_root: str | Path) -> tuple[list[dict], dict[str, int]]:
    """Per-prompt published records for every cell under ``cells_root``.

    A cell directory is ``<split>/<model>/<precision>``; each holds one
    ``gen_<task>.jsonl`` per task and, for judged tasks, ``judge_<task>.jsonl``.
    Returns the records and counts of cells, tasks and records.
    """
    root = Path(cells_root)
    rows: list[dict] = []
    cells: set[Path] = set()
    tasks = 0
    for gen in sorted(root.rglob("gen_*.jsonl")):
        cell = gen.parent
        cells.add(cell)
        tasks += 1
        jpath = cell / f"judge_{gen.stem[len('gen_'):]}.jsonl"
        verdicts = {v["id"]: v for v in _jsonl(jpath)} if jpath.exists() else {}
        for g in _jsonl(gen):
            rec = {"model": cell.parent.name, "precision": cell.name,
                   "eval_split": cell.parent.parent.name,
                   **{k: g[k] for k in GEN_KEEP if k in g}}
            v = verdicts.get(g.get("id"))
            if v:
                rec.update({f"judge_{k}": v[k] for k in JUDGE_KEEP if k in v and k != "id"})
            rows.append(rec)
    return rows, {"cells": len(cells), "tasks": tasks, "records": len(rows)}


def build_results_release(cells_root: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Write ``per_prompt_judgments.jsonl``, refusing to publish response text."""
    rows, stats = collect_records(cells_root)
    leaked = sorted(BANNED & {k for r in rows for k in r})
    if leaked:
        raise RuntimeError(f"refusing to publish: response-bearing fields present {leaked}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_prompt_judgments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8")
    return stats
