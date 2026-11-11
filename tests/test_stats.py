"""Tests for the paired statistics. Reference values computed by hand or from
standard tables; each guards a number that would appear in the write-up."""

from __future__ import annotations

import math

import pytest

from refusals_that_stick.stats import (binom_two_sided, bootstrap_mean_ci, holm, mcnemar_exact,
                                       paired_binary, ranks, spearman, wilson)


def test_binom_symmetric_known_value():
    # n=10, k=1 at p=0.5: P(X<=1)+P(X>=9) = 2*(1+10)/1024 = 0.021484375
    assert binom_two_sided(1, 10) == pytest.approx(22 / 1024)


def test_binom_center_is_one():
    assert binom_two_sided(5, 10) == pytest.approx(1.0)


def test_binom_empty():
    assert binom_two_sided(0, 0) == 1.0


def test_mcnemar_no_discordance_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric_in_arguments():
    assert mcnemar_exact(2, 9) == pytest.approx(mcnemar_exact(9, 2))


def test_mcnemar_known_value():
    # b=0, c=6: two-sided exact p = 2 * 0.5**6 = 0.03125
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)


def test_wilson_contains_point_and_is_bounded():
    lo, hi = wilson(10, 100)
    assert 0.0 <= lo < 0.10 < hi <= 1.0
    # Known reference: 10/100 -> (0.0552, 0.1744)
    assert lo == pytest.approx(0.0552, abs=5e-4)
    assert hi == pytest.approx(0.1744, abs=5e-4)


def test_wilson_zero_successes_has_positive_upper():
    lo, hi = wilson(0, 150)
    assert lo == 0.0
    assert 0.0 < hi < 0.03


def test_holm_reference():
    # Classic example: p = [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06]
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_is_monotone_and_capped():
    adj = holm([0.5, 0.6, 0.9])
    assert all(a <= 1.0 for a in adj)
    assert adj == pytest.approx([1.0, 1.0, 1.0])


def test_paired_binary_counts():
    a = {"p1": False, "p2": False, "p3": True, "p4": False}
    b = {"p1": True, "p2": True, "p3": True, "p4": False, "extra": True}
    r = paired_binary(a, b)
    assert r["n"] == 4  # "extra" is unpaired and ignored
    assert r["rate_a"] == pytest.approx(0.25)
    assert r["rate_b"] == pytest.approx(0.75)
    assert r["delta"] == pytest.approx(0.5)
    assert (r["discordant_a_only"], r["discordant_b_only"]) == (0, 2)
    assert r["p_mcnemar"] == pytest.approx(0.5)
    assert math.isfinite(r["ci_a"][0])


# --- ranks and Spearman ---------------------------------------------------------

def test_ranks_no_ties():
    assert ranks([10.0, 30.0, 20.0]) == [1.0, 3.0, 2.0]


def test_ranks_ties_share_average():
    # Two values tied for ranks 2 and 3 both get 2.5.
    assert ranks([1.0, 5.0, 5.0, 9.0]) == [1.0, 2.5, 2.5, 4.0]


def test_ranks_all_tied():
    assert ranks([7.0, 7.0, 7.0]) == [2.0, 2.0, 2.0]


def test_spearman_perfect_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_rank_based_not_linear():
    # A strictly increasing but wildly non-linear map still gives exactly 1.
    assert spearman([1, 2, 3, 4, 5], [1, 4, 9, 1000, 1e6]) == pytest.approx(1.0)


def test_spearman_known_value():
    # x=[1,2,3,4,5], y=[2,1,4,3,5] -> one swapped pair at each end.
    # d = [-1, 1, -1, 1, 0], sum d^2 = 4, rho = 1 - 6*4/(5*24) = 0.8
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)


def test_spearman_no_variation_is_none():
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_spearman_too_few_points():
    assert spearman([1, 2], [2, 1]) is None


def test_spearman_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        spearman([1, 2, 3], [1, 2])


class TestBootstrapMeanCI:
    """The interval behind the one headline a ratio cannot carry honestly."""

    def test_constant_values_give_a_degenerate_interval(self):
        r = bootstrap_mean_ci([0.2] * 50, iters=200, seed=1)
        assert r["mean"] == pytest.approx(0.2)
        assert r["lo"] == pytest.approx(0.2) and r["hi"] == pytest.approx(0.2)
        assert r["excludes_zero"] is True

    def test_interval_brackets_the_mean_and_is_seeded(self):
        values = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0] * 10
        a = bootstrap_mean_ci(values, iters=500, seed=7)
        b = bootstrap_mean_ci(values, iters=500, seed=7)
        c = bootstrap_mean_ci(values, iters=500, seed=8)
        assert a == b                      # reproducible, so a published interval is checkable
        assert a["lo"] <= a["mean"] <= a["hi"]
        assert (c["lo"], c["hi"]) != (a["lo"], a["hi"])

    def test_more_prompts_narrow_the_interval(self):
        small = bootstrap_mean_ci([1, 0] * 25, iters=400, seed=3)
        large = bootstrap_mean_ci([1, 0] * 500, iters=400, seed=3)
        assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])

    def test_zero_mean_interval_includes_zero(self):
        # Equal numbers of +1 and -1: quantization took back as much as it gave.
        r = bootstrap_mean_ci([1, -1] * 100, iters=400, seed=2)
        assert r["mean"] == pytest.approx(0.0)
        assert r["lo"] < 0 < r["hi"]
        assert r["excludes_zero"] is False

    def test_wider_level_gives_a_wider_interval(self):
        values = [1, 0, 0, 1, 1, 0, 0, 0] * 20
        narrow = bootstrap_mean_ci(values, iters=1000, seed=5, level=0.95)
        wide = bootstrap_mean_ci(values, iters=1000, seed=5, level=0.9958)
        assert (wide["hi"] - wide["lo"]) >= (narrow["hi"] - narrow["lo"])

    def test_empty_input_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError):
            bootstrap_mean_ci([])
