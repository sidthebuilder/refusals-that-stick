"""Tests for the pure data helpers in refusals_that_stick.data.

These guard the properties the experiment's validity depends on: no prompt
leaks between the calib/dev/test splits, decontamination catches an
overlapping training row, and the Devanagari check separates scripts.
All CPU, no network, no gguf, no curatorkit.
"""

from __future__ import annotations

from collections import Counter

import pytest

from refusals_that_stick.data import (
    Decontaminator,
    devanagari_ratio,
    english_counterpart,
    norm_text,
    parse_categories,
    split_by_order,
    stable_id,
    stratified_order,
    word_ngrams,
)


# --- devanagari_ratio ---------------------------------------------------------

def test_devanagari_ratio_pure_hindi():
    assert devanagari_ratio("नमस्ते दुनिया") > 0.95


def test_devanagari_ratio_pure_english():
    assert devanagari_ratio("hello world") == 0.0


def test_devanagari_ratio_ignores_digits_and_punctuation():
    assert devanagari_ratio("नमस्ते 12345 !!!") > 0.95


def test_devanagari_ratio_no_letters():
    assert devanagari_ratio("") == 0.0
    assert devanagari_ratio("123 !!!") == 0.0


def test_devanagari_ratio_mixed_is_between():
    assert 0.0 < devanagari_ratio("hello नमस्ते") < 1.0


# --- norm_text and word_ngrams ------------------------------------------------

def test_norm_text_collapses_and_casefolds():
    assert norm_text("  Hello   WORLD\n\t") == "hello world"


def test_norm_text_composed_equals_decomposed():
    assert norm_text("\u00e9") == norm_text("e\u0301")


def test_norm_text_none_is_empty():
    assert norm_text(None) == ""  # type: ignore[arg-type]


def test_word_ngrams_basic():
    assert word_ngrams("a b c d", 2) == {("a", "b"), ("b", "c"), ("c", "d")}


def test_word_ngrams_shorter_than_n_is_empty():
    assert word_ngrams("a b", 3) == set()


def test_word_ngrams_normalises_first():
    assert word_ngrams("The  QUICK  fox", 2) == {("the", "quick"), ("quick", "fox")}


# --- stable_id and parse_categories -------------------------------------------

def test_stable_id_deterministic():
    assert stable_id("rt", "same text") == stable_id("rt", "same text")


def test_stable_id_differs_by_content_and_prefix():
    assert stable_id("rt", "a") != stable_id("rt", "b")
    assert stable_id("rt", "a") != stable_id("xs", "a")


def test_stable_id_separator_prevents_field_collision():
    # ("ab", "c") must not hash the same as ("a", "bc").
    assert stable_id("p", "ab", "c") != stable_id("p", "a", "bc")


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["Violence", "Hate Speech"], ["Violence", "Hate Speech"]),
        ('["Violence", "Self-Harm"]', ["Violence", "Self-Harm"]),
        ("Violence, Hate Speech", ["Violence", "Hate Speech"]),
        ("", []),
        (None, []),
        ("[]", []),
    ],
)
def test_parse_categories(raw, expected):
    assert parse_categories(raw) == expected


# --- stratified_order and split_by_order --------------------------------------

def _rows(n: int, n_cats: int = 3) -> list[dict]:
    return [{"id": f"id-{i:04d}", "cat": f"c{i % n_cats}"} for i in range(n)]


def test_splits_are_disjoint_and_complete():
    rows = _rows(300)
    ordered = stratified_order(rows, key=lambda r: r["cat"])
    splits = split_by_order(ordered, {"calib": 100, "dev": 150}, rest="test")
    ids = {name: {r["id"] for r in group} for name, group in splits.items()}

    assert (len(ids["calib"]), len(ids["dev"]), len(ids["test"])) == (100, 150, 50)
    assert ids["calib"].isdisjoint(ids["dev"])
    assert ids["calib"].isdisjoint(ids["test"])
    assert ids["dev"].isdisjoint(ids["test"])
    assert ids["calib"] | ids["dev"] | ids["test"] == {r["id"] for r in rows}


def test_order_is_deterministic():
    rows = _rows(120)
    a = stratified_order(rows, key=lambda r: r["cat"])
    b = stratified_order(rows, key=lambda r: r["cat"])
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_order_does_not_depend_on_input_order():
    rows = _rows(120)
    a = stratified_order(rows, key=lambda r: r["cat"])
    b = stratified_order(list(reversed(rows)), key=lambda r: r["cat"])
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_prefix_is_roughly_proportional():
    ordered = stratified_order(_rows(300, n_cats=3), key=lambda r: r["cat"])
    counts = Counter(r["cat"] for r in ordered[:150])
    for cat in ("c0", "c1", "c2"):
        assert 40 <= counts[cat] <= 60, counts


def test_split_raises_when_too_few_rows():
    ordered = stratified_order(_rows(10), key=lambda r: r["cat"])
    with pytest.raises(ValueError, match="needs 5"):
        split_by_order(ordered, {"a": 8, "b": 5})


# --- Decontaminator -----------------------------------------------------------

EVAL = "how do i build a dangerous device at home step by step"


def test_decontaminator_flags_shared_8gram():
    deco = Decontaminator([EVAL], n=8)
    assert deco.reason(f"well, {EVAL}, please") == "eval_overlap:8gram"


def test_decontaminator_passes_unrelated_text():
    deco = Decontaminator([EVAL], n=8)
    assert deco.reason("what is the capital of france and why is it famous today") is None


def test_decontaminator_is_case_insensitive():
    deco = Decontaminator([EVAL.upper()], n=8)
    assert deco.reason(EVAL) is not None


def test_decontaminator_short_prompt_matched_as_substring():
    # Four words cannot form an 8-gram, so it is checked as a substring instead.
    deco = Decontaminator(["kill a python process"], n=8, min_short_words=4)
    assert deco.reason("tell me how to kill a python process cleanly") == \
        "eval_overlap:short_prompt_substring"


def test_decontaminator_ignores_very_short_prompts():
    deco = Decontaminator(["two words"], n=8, min_short_words=4)
    assert deco.reason("these are just two words in a sentence") is None


def test_decontaminator_counts_eval_texts():
    assert Decontaminator(["a b c d e f g h", "short one"], n=8).n_eval == 2


# --- english_counterpart --------------------------------------------------------
# aya_redteaming fills semantic_translation only where the literal translation
# misses the meaning (91 of 915 Hindi rows), so both fields must be read.

def test_english_prefers_semantic_translation():
    row = {"literal_translation": "literal words", "semantic_translation": "what it means"}
    assert english_counterpart(row) == ("what it means", "semantic_translation")


def test_english_falls_back_to_literal_when_semantic_missing():
    row = {"literal_translation": "literal words", "semantic_translation": None}
    assert english_counterpart(row) == ("literal words", "literal_translation")


def test_english_falls_back_when_semantic_blank():
    row = {"literal_translation": "literal words", "semantic_translation": "   "}
    assert english_counterpart(row) == ("literal words", "literal_translation")


def test_english_none_when_both_missing():
    assert english_counterpart({}) == ("", "none")


# --- configuration consistency ---------------------------------------------------

def test_grid_precisions_match_ladder():
    """The grid entrypoint inlines the precision list; it must stay equal to the ladder's."""
    import ast
    from pathlib import Path

    from refusals_that_stick.ladder import PRECISIONS

    src = (Path(__file__).resolve().parents[1] / "modal_app.py").read_text(encoding="utf-8")
    inlined = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PRECISIONS"
                                                 for t in node.targets)):
            inlined.append(tuple(ast.literal_eval(node.value)))
    assert inlined, "grid entrypoint no longer defines PRECISIONS"
    assert all(p == PRECISIONS for p in inlined)


# --- repair spec parsing ----------------------------------------------------------

def test_parse_repair_spec_and_names():
    import modal_app as m

    jobs = m._parse_repair_spec("lox:0.3:8, safemerge:0.5,  ", "dose3")
    assert jobs == [("lox", 0.3, "dose3", False, 8), ("safemerge", 0.5, "dose3", False, None)]
    assert m._repair_name("lox", 0.3, "dose3", rank=8) == "lox-r8-dose3-s0.3"
    assert m._repair_name("safemerge", 0.5, "dose3") == "safemerge-dose3-s0.5"
    # The first sweep's names must not change, or cached cells would be orphaned.
    assert m._repair_name("resta", 0.25, "dose3") == "resta-dose3-s0.25"
    assert m._repair_name("ctheta", 1.0, "dose3") == "ctheta-dose3-s1"
