"""Tests for the paired statistics. Reference values computed by hand or from
standard tables; each guards a number that would appear in the write-up."""

from __future__ import annotations

import math

import pytest

from refusals_that_stick.stats import (binom_two_sided, holm, mcnemar_exact, paired_binary,
                                       wilson)


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
