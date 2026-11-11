"""Tests for the publication guard: labels go out, response text does not."""

from __future__ import annotations

import json

import pytest

from refusals_that_stick import release
from refusals_that_stick.release import BANNED, build_results_release, collect_records


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


@pytest.fixture()
def cells(tmp_path):
    """Two cells of one model, as the grid writes them: <split>/<model>/<precision>."""
    root = tmp_path / "cells"
    for precision in ("BF16", "Q4_K_M"):
        cell = root / "dev" / "drift-dose3" / precision
        _write(cell / "gen_redteam_hi_dev.jsonl", [
            {"id": "a", "item": "item001", "lang": "hi", "task": "redteam_hi_dev",
             "prompt": "<harmful request>", "response": "<harmful compliance>",
             "degenerate": None, "completion_tokens": 42, "devanagari_ratio": 0.9},
            {"id": "b", "item": "item002", "lang": "hi", "task": "redteam_hi_dev",
             "prompt": "<harmful request>", "response": "I can't help with that",
             "degenerate": None, "completion_tokens": 7, "devanagari_ratio": 0.1},
        ])
        _write(cell / "judge_redteam_hi_dev.jsonl", [
            {"id": "a", "parsed": True, "refusal": False, "safety": "Unsafe",
             "categories": ["Violent"], "raw": "Safety: Unsafe\nRefusal: No"},
            {"id": "b", "parsed": True, "refusal": True, "safety": "Safe",
             "categories": [], "raw": "Safety: Safe\nRefusal: Yes"},
        ])
    return root


def test_no_response_or_prompt_text_is_staged(cells, tmp_path):
    stats = build_results_release(cells, tmp_path / "out")
    assert stats == {"cells": 2, "tasks": 2, "records": 4}
    text = (tmp_path / "out" / "per_prompt_judgments.jsonl").read_text(encoding="utf-8")
    assert "harmful request" not in text
    assert "harmful compliance" not in text
    assert "I can't help with that" not in text
    rows = [json.loads(l) for l in text.splitlines()]
    assert not BANNED & {k for r in rows for k in r}


def test_labels_and_cell_coordinates_survive(cells, tmp_path):
    build_results_release(cells, tmp_path / "out")
    rows = [json.loads(l) for l in
            (tmp_path / "out" / "per_prompt_judgments.jsonl").read_text(
                encoding="utf-8").splitlines()]
    bf16 = {r["id"]: r for r in rows if r["precision"] == "BF16"}
    assert {r["model"] for r in rows} == {"drift-dose3"}
    assert {r["eval_split"] for r in rows} == {"dev"}
    assert bf16["a"]["judge_refusal"] is False and bf16["a"]["judge_safety"] == "Unsafe"
    assert bf16["b"]["judge_refusal"] is True
    assert bf16["a"]["completion_tokens"] == 42  # metadata needed for the degeneracy audit


def test_guard_refuses_when_a_banned_field_slips_into_the_allow_list(cells, tmp_path,
                                                                    monkeypatch):
    """The allow-list is the mechanism; the guard is the backstop if it is widened."""
    monkeypatch.setattr(release, "GEN_KEEP", release.GEN_KEEP + ("response",))
    with pytest.raises(RuntimeError, match="response-bearing fields present"):
        build_results_release(cells, tmp_path / "out")
    assert not (tmp_path / "out" / "per_prompt_judgments.jsonl").exists()


def test_judge_raw_text_is_never_copied(cells, tmp_path):
    rows, _ = collect_records(cells)
    assert all("judge_raw" not in r for r in rows)


def test_missing_judge_file_leaves_generation_metadata(tmp_path):
    cell = tmp_path / "cells" / "dev" / "fire" / "BF16"
    _write(cell / "gen_mmlu_hi_dev.jsonl",
           [{"id": "m1", "answer": "C", "choice": "C", "correct": True,
             "response": "उत्तर: सी", "degenerate": None}])
    rows, stats = collect_records(tmp_path / "cells")
    assert stats == {"cells": 1, "tasks": 1, "records": 1}
    assert rows[0]["answer"] == "C" and rows[0]["choice"] == "C"
    assert "response" not in rows[0] and "judge_refusal" not in rows[0]
