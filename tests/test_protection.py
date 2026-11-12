"""Tests for computing mixed-precision protection from per-tensor survival maps."""

from __future__ import annotations

import pytest

from refusals_that_stick.protection import (BITS_PER_WEIGHT, circuit_tensor_names,
                                            pooled_survival, protected_survival,
                                            random_matched_set, size_bytes)


def _row(name, n_edited, perm, dirp, n_params=100, qtype="Q4_K"):
    return {"name": name, "n_edited": n_edited, "n_params": n_params, "qtype": qtype,
            "permanence": perm, "directional_permanence": dirp, "erased_fraction": 1 - perm}


BASE = [_row("a", 60, 1.0, 0.5), _row("b", 40, 0.5, 0.25), {"name": "c", "n_edited": 0,
                                                            "n_params": 100}]
Q8 = [_row("a", 60, 0.9, 0.9, qtype="Q8_0"), _row("b", 40, 0.8, 0.8, qtype="Q8_0")]


def test_pooling_weights_by_edited_count_and_skips_untouched():
    p = pooled_survival(BASE)
    assert p["n_edited"] == 100
    assert p["permanence"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)
    assert p["directional_permanence"] == pytest.approx(0.6 * 0.5 + 0.4 * 0.25)
    assert p["direction_among_moved"] == pytest.approx(0.4 / 0.8)


def test_protecting_nothing_equals_the_base_file():
    assert protected_survival(BASE, Q8, [])["directional_permanence"] == pytest.approx(
        pooled_survival(BASE)["directional_permanence"])


def test_protecting_every_edited_tensor_equals_the_q8_file():
    out = protected_survival(BASE, Q8, ["a", "b"])
    assert out["directional_permanence"] == pytest.approx(pooled_survival(Q8)["directional_permanence"])
    assert out["n_edited_tensors_protected"] == 2


def test_protecting_an_untouched_tensor_changes_nothing():
    out = protected_survival(BASE, Q8, ["c"])
    assert out["directional_permanence"] == pytest.approx(pooled_survival(BASE)["directional_permanence"])
    assert out["n_edited_tensors_protected"] == 0


def test_partial_protection_mixes_rows():
    out = protected_survival(BASE, Q8, ["b"])
    assert out["directional_permanence"] == pytest.approx(0.6 * 0.5 + 0.4 * 0.8)


def test_size_counts_protected_tensors_at_q8_and_never_retypes_f32():
    rows = [{"name": "w", "n_params": 256, "qtype": "Q4_K"},
            {"name": "norm", "n_params": 32, "qtype": "F32"}]
    assert size_bytes(rows) == pytest.approx(256 * 4.5 / 8 + 32 * 4)
    assert size_bytes(rows, ["w", "norm"]) == pytest.approx(256 * 8.5 / 8 + 32 * 4)
    assert BITS_PER_WEIGHT["Q8_0"] == pytest.approx(8.5)


def test_circuit_names_map_projections_to_gguf():
    names = circuit_tensor_names([14, 0], ["q_proj", "down_proj"])
    assert names == ["blk.0.attn_q.weight", "blk.0.ffn_down.weight",
                     "blk.14.attn_q.weight", "blk.14.ffn_down.weight"]
    with pytest.raises(ValueError):
        circuit_tensor_names([0], ["lm_head"])


def test_random_set_is_seeded_matched_and_skips_f32():
    rows = [{"name": f"t{i}", "n_params": 10, "qtype": "Q4_K"} for i in range(50)]
    rows.append({"name": "norm", "n_params": 10, "qtype": "F32"})
    a = random_matched_set(rows, 100, seed=1)
    assert a == random_matched_set(rows, 100, seed=1)
    assert a != random_matched_set(rows, 100, seed=2)
    assert "norm" not in a
    assert sum(10 for _ in a) == 100


def test_random_set_reports_an_unreachable_target():
    rows = [{"name": "big", "n_params": 1000, "qtype": "Q4_K"}]
    with pytest.raises(RuntimeError):
        random_matched_set(rows, 100, seed=0)
