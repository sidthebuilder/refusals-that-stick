"""Tests for the GGUF survival measurement.

These run on CPU in seconds and validated the project's core mechanism before
any GPU budget was spent. Three of them record findings that cost nothing here
and would have been expensive or misleading later:

* ``test_kquants_cannot_be_simulated`` — the ``gguf`` package cannot quantize to
  k-quants, so those must be measured from real llama-quantize output.
* ``test_noise_is_not_mistaken_for_survival`` — a magnitude ratio scores pure
  quantization noise as better-than-perfect preservation.
* ``test_projection_inflates_when_realized_dwarfs_intended`` — the least-squares
  projection scores ~1.9 when 97% of the edit was in fact erased.

Together those forced the headline metric to be per-weight permanence.
"""

from __future__ import annotations

import numpy as np
import pytest

from refusals_that_stick.survival import (
    aggregate_retention,
    apply_magnitude_floor,
    block_size,
    can_roundtrip,
    compare_edits,
    effective_bin_width,
    quant_roundtrip,
    resolve_qtype,
    tensor_survival,
)

# Realistic transformer weight scale; last dim a multiple of 256 so k-quant
# block requirements are satisfied.
SHAPE = (64, 256)
WEIGHT_SD = 0.02
SEED = 0

# Types the gguf package can actually quantize to, so the simulated path works.
SIMULABLE = ["Q4_0", "Q8_0"]


def _weights(shape=SHAPE, sd=WEIGHT_SD, seed=SEED) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sd, size=shape).astype(np.float32)


def _sparse_edit(before, magnitude, *, fraction=0.02, seed=SEED + 1):
    """Edit a small share of weights by ``magnitude``, leaving the rest alone.

    This is the shape of a circuit-restricted repair: C-ΔΘ reports touching
    under 5% of parameters.
    """
    rng = np.random.default_rng(seed)
    delta = np.zeros_like(before)
    flat = delta.reshape(-1)
    n = max(1, int(flat.size * fraction))
    idx = rng.choice(flat.size, size=n, replace=False)
    flat[idx] = rng.choice([-1.0, 1.0], size=n) * magnitude
    return (before + delta).astype(np.float32), delta


# --------------------------------------------------------------------------- #
# Format plumbing
# --------------------------------------------------------------------------- #

def test_resolve_qtype_accepts_name_and_enum():
    qt = resolve_qtype("Q4_K")
    assert resolve_qtype(qt) is qt
    assert resolve_qtype("q4_k") is qt


def test_resolve_qtype_rejects_typo_with_helpful_message():
    with pytest.raises(ValueError, match="valid names"):
        resolve_qtype("Q4_KM")


def test_block_sizes_match_ggml():
    assert block_size("Q4_0") == 32
    assert block_size("Q4_K") == 256


def test_kquants_cannot_be_simulated():
    """A real constraint: k-quants are dequantize-only in NumPy.

    This is why the project measures real GGUF files instead of simulating.
    """
    assert can_roundtrip("Q4_0") is True
    assert can_roundtrip("Q8_0") is True
    assert can_roundtrip("Q4_K") is False
    assert can_roundtrip("Q2_K") is False

    with pytest.raises(NotImplementedError, match="llama-quantize"):
        quant_roundtrip(_weights(), "Q4_K")


def test_shape_guard_explains_block_requirement():
    with pytest.raises(ValueError, match="blocks of 32"):
        quant_roundtrip(_weights(shape=(8, 100)), "Q4_0")


def test_shape_and_dtype_preserved():
    before = _weights()
    out = quant_roundtrip(before, "Q4_0")
    assert out.shape == before.shape
    assert out.dtype == np.float32


# --------------------------------------------------------------------------- #
# Metric behaviour on hand-built cases
# --------------------------------------------------------------------------- #

def test_exact_edit_scores_one():
    intended = np.array([0.0, 0.5, -0.25, 0.1])
    r = compare_edits(intended, intended.copy())
    assert r.permanence == pytest.approx(1.0)
    assert r.directional_permanence == pytest.approx(1.0)
    assert r.aligned_retention == pytest.approx(1.0)
    assert r.noise_ratio == pytest.approx(0.0, abs=1e-12)


def test_fully_erased_edit_scores_zero():
    intended = np.array([0.0, 0.5, -0.25, 0.1])
    r = compare_edits(intended, np.zeros_like(intended))
    assert r.permanence == pytest.approx(0.0)
    assert r.directional_permanence == pytest.approx(0.0)
    assert r.erased_fraction == pytest.approx(1.0)


def test_reversed_edit_is_distinguished_from_erasure():
    """A reversed edit moved, so permanence is 1, but direction is 0."""
    intended = np.array([0.5, -0.25])
    r = compare_edits(intended, -intended)
    assert r.permanence == pytest.approx(1.0)
    assert r.directional_permanence == pytest.approx(0.0)
    assert r.aligned_retention == pytest.approx(-1.0)


def test_noise_is_not_mistaken_for_survival():
    """A magnitude ratio fails here; the per-weight metric does not."""
    rng = np.random.default_rng(7)
    n = 20000
    intended = rng.choice([-1.0, 1.0], size=n) * 1e-4
    noise = rng.normal(0.0, 3e-4, size=n)  # larger in magnitude than the edit

    r = compare_edits(intended, noise)

    assert r.magnitude_ratio > 1.0, "setup: noise should dominate in magnitude"
    assert r.directional_permanence < 0.6, "noise must not read as directed survival"
    assert abs(r.aligned_retention) < 0.05
    assert r.noise_ratio > 1.0


def test_projection_inflates_when_realized_dwarfs_intended():
    """Why the projection is a diagnostic, not the headline.

    Most of the edit is erased, but the few weights that move jump far further
    than asked. The projection reads well above 1.0; permanence correctly
    reports that almost nothing survived.
    """
    n = 1000
    intended = np.full(n, 1e-5)
    realized = np.zeros(n)
    realized[:30] = 1e-3  # 3% of weights jump ~100x the intended amount

    r = compare_edits(intended, realized)

    assert r.aligned_retention > 1.5, "setup: projection should be inflated here"
    assert r.permanence == pytest.approx(0.03)
    assert r.directional_permanence == pytest.approx(0.03)


def test_untouched_tensor_reports_full_retention():
    before = _weights()
    r = tensor_survival(before, before.copy(), "Q4_0", name="untouched")
    assert r.n_edited == 0
    assert r.directional_permanence == 1.0


# --------------------------------------------------------------------------- #
# The core hypothesis, on simulable formats
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qtype", SIMULABLE)
def test_sparse_sub_bin_edit_does_not_survive(qtype):
    """The mechanism behind the project's main hypothesis.

    A sparse edit far below one quantization bin is invisible per weight in the
    deployed model: each touched weight rounds back to where it started.
    """
    before = _weights()
    bin_w = effective_bin_width(before, qtype)
    assert bin_w > 0, "a lossy format must have a measurable bin width"

    after, _ = _sparse_edit(before, bin_w / 100.0)
    r = tensor_survival(before, after, qtype, name=f"sparse-tiny@{qtype}")

    assert r.directional_permanence < 0.25, (
        f"{qtype}: expected erasure, got directional permanence "
        f"{r.directional_permanence:.3f}"
    )
    assert r.erased_fraction > 0.5


@pytest.mark.parametrize("qtype", SIMULABLE)
def test_sparse_large_edit_survives(qtype):
    """An edit far above one bin must come through, or the metric is broken."""
    before = _weights()
    bin_w = effective_bin_width(before, qtype)
    after, _ = _sparse_edit(before, bin_w * 20.0, seed=SEED + 2)

    r = tensor_survival(before, after, qtype, name=f"sparse-big@{qtype}")

    assert r.directional_permanence > 0.9, (
        f"{qtype}: large edit should survive, got {r.directional_permanence:.3f}"
    )
    assert r.permanence > 0.9
    assert r.cosine > 0.9


@pytest.mark.parametrize("qtype", SIMULABLE)
def test_dense_sub_bin_edit_also_fails_per_weight(qtype):
    """Documents that erasure is not merely about sparsity.

    When every weight is nudged below one bin, the aggregate sum can look
    preserved because block scales shift and weights flip bins stochastically.
    Per weight, though, the edit is still not there: direction agreement sits
    near chance and the realized change is dominated by noise. This is why the
    headline metric is per weight rather than summed.
    """
    before = _weights()
    bin_w = effective_bin_width(before, qtype)
    rng = np.random.default_rng(SEED + 1)
    tiny = bin_w / 100.0
    after = before + rng.choice([-tiny, tiny], size=before.shape).astype(np.float32)

    r = tensor_survival(before, after, qtype, name=f"dense-tiny@{qtype}")

    assert r.directional_permanence < 0.6
    assert r.noise_ratio > 1.0


# --------------------------------------------------------------------------- #
# The proposed fix
# --------------------------------------------------------------------------- #

def test_magnitude_floor_rescues_a_sparse_sub_bin_edit():
    """The fix must raise survival on exactly the case that fails."""
    qtype = "Q4_0"
    before = _weights()
    bin_w = effective_bin_width(before, qtype)
    after, delta = _sparse_edit(before, bin_w / 100.0, seed=SEED + 3)

    plain = tensor_survival(before, after, qtype)
    floored = apply_magnitude_floor(before, after, qtype, multiplier=1.0)
    fixed = tensor_survival(before, floored, qtype)

    assert plain.directional_permanence < 0.25, "setup: the plain edit should be erased"
    assert fixed.directional_permanence > plain.directional_permanence
    assert fixed.erased_fraction < plain.erased_fraction

    # The floor must preserve the direction of every edit it touches.
    touched = delta != 0
    assert np.all(np.sign((floored - before)[touched]) == np.sign(delta[touched]))


def test_magnitude_floor_leaves_large_edits_alone():
    qtype = "Q4_0"
    before = _weights()
    after = before + effective_bin_width(before, qtype) * 50.0
    np.testing.assert_allclose(
        apply_magnitude_floor(before, after, qtype), after, rtol=0, atol=1e-6
    )


def test_magnitude_floor_accepts_explicit_bin_width_for_kquants():
    """K-quants cannot be measured in NumPy, so the width can be supplied.

    The delta must stay above float32 resolution at this weight scale, or it
    rounds to exactly zero and there is nothing to lift.
    """
    before = _weights()
    after = before + 1e-5
    out = apply_magnitude_floor(before, after, "Q4_K", bin_width=1e-3)
    np.testing.assert_allclose(np.abs(out - before), 1e-3, rtol=1e-4)


def test_magnitude_floor_rejects_bad_multiplier():
    before = _weights()
    with pytest.raises(ValueError, match="positive"):
        apply_magnitude_floor(before, before + 1.0, "Q4_0", multiplier=0.0)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_aggregate_weights_by_edit_count():
    """A heavily edited tensor must dominate a barely edited one."""
    erased = compare_edits(np.full(10, 1e-6), np.zeros(10), name="erased")
    survived = compare_edits(np.full(990, 1.0), np.full(990, 1.0), name="survived")
    pooled = aggregate_retention([erased, survived])
    assert pooled == pytest.approx(0.99)


def test_aggregate_ignores_untouched_tensors():
    touched = compare_edits(np.full(10, 1.0), np.zeros(10))
    untouched = compare_edits(np.zeros(10), np.zeros(10))
    assert aggregate_retention([touched, untouched]) == pytest.approx(0.0)


def test_aggregate_can_pool_other_fields():
    a = compare_edits(np.full(10, 1.0), np.full(10, 1.0))
    b = compare_edits(np.full(10, 1.0), np.zeros(10))
    assert aggregate_retention([a, b], field="permanence") == pytest.approx(0.5)
