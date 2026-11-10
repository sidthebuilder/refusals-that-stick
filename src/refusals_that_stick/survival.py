"""Measure how much of a weight-space edit survives GGUF quantization.

The question: a safety repair (SafeTune C-ΔΘ, RESTA, ...) changes a model's
weights. Deployment quantizes them. Does the repair survive?

MANSU (arXiv:2605.15138) showed that for *unlearning*, 4-bit quantization can
erase an edit entirely: a per-weight update smaller than one quantization bin
rounds back to the original value. Safety repairs are also small, sparse edits
(C-ΔΘ reports touching under 5% of parameters), so the same failure mode is
possible. Nobody has measured it for repairs. This module measures it.


Choosing the metric (two wrong answers first)
---------------------------------------------
Both obvious measures fail, and our own tests are what showed it.

*Magnitude ratio*, ``sum|realized| / sum|intended|``: quantizing two nearly
identical checkpoints independently re-estimates block scales, so weights jump
between bins in ways uncorrelated with the edit. On a sub-bin edit this scored
**1.41** on Q4_0 while sign agreement sat at chance. Pure noise read as
better-than-perfect preservation.

*Least-squares projection*, ``<realized, intended> / <intended, intended>``:
robust to uncorrelated noise, but inflates badly when the realized changes are
far larger than the intended ones. On a sparse sub-bin edit at Q8_0 it scored
**1.93** while 97% of the edited weights had in fact rounded straight back: the
3% that moved jumped a whole bin, and those few large terms dominated the dot
product.

So the headline is per-weight **permanence**, which is what MANSU actually
defines (Lemma 1: ``Q(θ_i + Δθ_i) ≠ Q(θ_i)`` for every edited ``i``):

    permanence             share of edited weights whose deployed value moved
    directional_permanence share that moved in the intended direction

These are bounded in [0, 1], interpretable, and immune to both failure modes
above. The projection and cosine are kept as diagnostics, with the caveat that
``aligned_retention`` may exceed 1.0.


Two measurement paths
---------------------
``from_gguf_pair`` is the real one: read two GGUF files produced by
``llama-quantize`` and compare dequantized tensors. This measures the actual
deployed artifact, so there is no simulation-fidelity question.

``quant_roundtrip`` simulates in NumPy, but the ``gguf`` package only implements
*quantization* for the legacy types (Q4_0, Q8_0, ...); k-quants such as Q4_K
can only be dequantized. ``can_roundtrip`` reports which is which, and the
simulation path is used for fast pre-checks and unit tests only.

All CPU, no GPU, no network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize, quantize

__all__ = [
    "TensorSurvival",
    "block_size",
    "resolve_qtype",
    "can_roundtrip",
    "quant_roundtrip",
    "effective_bin_width",
    "compare_edits",
    "tensor_survival",
    "apply_magnitude_floor",
    "aggregate_retention",
    "read_gguf_tensors",
    "from_gguf_pair",
]


def resolve_qtype(qtype: str | GGMLQuantizationType) -> GGMLQuantizationType:
    """Accept ``"Q4_K"`` or the enum and return the enum."""
    if isinstance(qtype, GGMLQuantizationType):
        return qtype
    key = str(qtype).strip().upper()
    try:
        return GGMLQuantizationType[key]
    except KeyError:
        valid = ", ".join(sorted(t.name for t in GGMLQuantizationType))
        raise ValueError(f"unknown GGUF quant type {qtype!r}; valid names: {valid}") from None


def block_size(qtype: str | GGMLQuantizationType) -> int:
    """Weights per quantization block."""
    return GGML_QUANT_SIZES[resolve_qtype(qtype)][0]


def can_roundtrip(qtype: str | GGMLQuantizationType) -> bool:
    """Whether NumPy can *quantize* to this type, not merely dequantize it.

    The ``gguf`` package implements quantization only for the legacy formats.
    K-quants (Q4_K, Q3_K, Q2_K, ...) raise ``NotImplementedError``, so they must
    be measured from real files produced by ``llama-quantize``.
    """
    qt = resolve_qtype(qtype)
    probe = np.zeros((1, GGML_QUANT_SIZES[qt][0]), dtype=np.float32)
    try:
        quantize(probe, qt)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - any other failure is a real problem, surface it
        raise
    return True


def _check_shape(arr: np.ndarray, qt: GGMLQuantizationType) -> None:
    bs = GGML_QUANT_SIZES[qt][0]
    if arr.ndim == 0:
        raise ValueError("expected an array with at least one dimension")
    if arr.shape[-1] % bs != 0:
        raise ValueError(
            f"{qt.name} quantizes in blocks of {bs}, so the last dimension must be a "
            f"multiple of {bs}; got shape {arr.shape}."
        )


def quant_roundtrip(weights: np.ndarray, qtype: str | GGMLQuantizationType) -> np.ndarray:
    """Quantize then dequantize in NumPy. Legacy types only."""
    qt = resolve_qtype(qtype)
    arr = np.ascontiguousarray(weights, dtype=np.float32)
    _check_shape(arr, qt)
    try:
        return dequantize(quantize(arr, qt), qt).astype(np.float32).reshape(arr.shape)
    except NotImplementedError:
        raise NotImplementedError(
            f"the gguf package cannot quantize to {qt.name} (only dequantize it). "
            "Measure this type with from_gguf_pair() on files built by llama-quantize, "
            "or check can_roundtrip() first."
        ) from None


def effective_bin_width(weights: np.ndarray, qtype: str | GGMLQuantizationType) -> float:
    """Estimate one quantization step for this tensor, empirically.

    Measured rather than derived, because the k-quants use block-wise scales
    plus minimums and a second level of scale quantization, so a closed form per
    type would be its own source of error. Reconstruction error averages about a
    quarter of a bin under uniform rounding, hence the factor of four.

    Used to set the magnitude floor and for context, never as a headline number.
    """
    qt = resolve_qtype(qtype)
    arr = np.ascontiguousarray(weights, dtype=np.float32)
    err = np.abs(arr - quant_roundtrip(arr, qt))
    nonzero = err[err > 0]
    return float(np.median(nonzero) * 4.0) if nonzero.size else 0.0


@dataclass(frozen=True)
class TensorSurvival:
    """How much of one tensor's edit survived quantization.

    Headline fields are ``permanence`` and ``directional_permanence``, both in
    [0, 1]. The rest are diagnostics; note that ``aligned_retention`` can exceed
    1.0 when a few weights jump a full bin, so it must not be quoted alone (see
    the module docstring).
    """

    name: str
    qtype: str
    n_params: int
    n_edited: int
    # Headline, per-weight and bounded.
    permanence: float
    directional_permanence: float
    # Diagnostics.
    aligned_retention: float
    cosine: float
    noise_ratio: float
    intended_l1: float
    realized_l1: float
    magnitude_ratio: float
    erased_fraction: float
    median_abs_intended: float

    def as_dict(self) -> dict:
        return asdict(self)


def compare_edits(
    intended: np.ndarray,
    realized: np.ndarray,
    *,
    name: str = "",
    qtype: str = "",
    n_params: int | None = None,
) -> TensorSurvival:
    """Core math, shared by the simulated and real-file paths.

    ``intended`` is ``after - before`` in full precision; ``realized`` is the
    difference the deployed (quantized) weights actually express. Metrics are
    computed over the positions the edit touched, which is the question being
    asked: of what the repair changed, how much is still there?
    """
    vi_full = np.ascontiguousarray(intended, dtype=np.float64).ravel()
    vr_full = np.ascontiguousarray(realized, dtype=np.float64).ravel()
    if vi_full.shape != vr_full.shape:
        raise ValueError(f"shape mismatch: intended {vi_full.shape} vs realized {vr_full.shape}")

    total = int(n_params if n_params is not None else vi_full.size)
    touched = vi_full != 0.0
    n_edited = int(touched.sum())

    if n_edited == 0:
        # Nothing was asked for, so nothing was lost. Reported as fully
        # preserved by convention so untouched tensors do not drag an aggregate
        # down; aggregates weight by edit count and skip these anyway.
        return TensorSurvival(
            name=name, qtype=qtype, n_params=total, n_edited=0,
            permanence=1.0, directional_permanence=1.0,
            aligned_retention=1.0, cosine=1.0, noise_ratio=0.0,
            intended_l1=0.0, realized_l1=0.0, magnitude_ratio=1.0,
            erased_fraction=0.0, median_abs_intended=0.0,
        )

    vi = vi_full[touched]
    vr = vr_full[touched]

    ii = float(np.dot(vi, vi))
    ir = float(np.dot(vi, vr))
    aligned = ir / ii if ii > 0 else 0.0

    # Component of the realized change that is not the intended edit.
    residual = vr - aligned * vi
    noise_ratio = float(np.linalg.norm(residual) / np.linalg.norm(vi)) if ii > 0 else 0.0

    ni, nr = float(np.linalg.norm(vi)), float(np.linalg.norm(vr))
    cosine = ir / (ni * nr) if ni > 0 and nr > 0 else 0.0

    intended_l1 = float(np.abs(vi).sum())
    realized_l1 = float(np.abs(vr).sum())

    moved = vr != 0.0
    return TensorSurvival(
        name=name,
        qtype=qtype,
        n_params=total,
        n_edited=n_edited,
        permanence=float(moved.mean()),
        directional_permanence=float((np.sign(vr) * np.sign(vi) > 0).mean()),
        aligned_retention=aligned,
        cosine=cosine,
        noise_ratio=noise_ratio,
        intended_l1=intended_l1,
        realized_l1=realized_l1,
        magnitude_ratio=realized_l1 / intended_l1 if intended_l1 > 0 else 1.0,
        erased_fraction=float((~moved).mean()),
        median_abs_intended=float(np.median(np.abs(vi))),
    )


def tensor_survival(
    before: np.ndarray,
    after: np.ndarray,
    qtype: str | GGMLQuantizationType,
    *,
    name: str = "",
) -> TensorSurvival:
    """Simulated path: quantize both tensors in NumPy and compare. Legacy types only.

    Each tensor is quantized independently, matching deployment, where the
    drifted and repaired checkpoints are converted to GGUF separately and so get
    their own block scales.
    """
    qt = resolve_qtype(qtype)
    b = np.ascontiguousarray(before, dtype=np.float32)
    a = np.ascontiguousarray(after, dtype=np.float32)
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: before {b.shape} vs after {a.shape}")
    realized = quant_roundtrip(a, qt) - quant_roundtrip(b, qt)
    return compare_edits(a - b, realized, name=name, qtype=qt.name, n_params=b.size)


def apply_magnitude_floor(
    before: np.ndarray,
    after: np.ndarray,
    qtype: str | GGMLQuantizationType,
    *,
    multiplier: float = 1.0,
    bin_width: float | None = None,
) -> np.ndarray:
    """Scale sub-bin edits up so quantization cannot round them away.

    MANSU's magnitude floor (arXiv:2605.15138, Eq. 4), carried from unlearning to
    safety repair: where the intended change is non-zero but smaller than one
    quantization bin, rescale it to one bin and keep its direction. Larger
    changes are untouched.

    ``bin_width`` may be supplied for types NumPy cannot quantize (measure it on
    a legacy type of similar width, or from a real GGUF). The caller must check
    that the stronger edit does not raise over-refusal, the obvious way this fix
    could backfire.
    """
    b = np.ascontiguousarray(before, dtype=np.float32)
    a = np.ascontiguousarray(after, dtype=np.float32)
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: before {b.shape} vs after {a.shape}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")

    floor = (effective_bin_width(b, qtype) if bin_width is None else float(bin_width)) * multiplier
    if floor <= 0:
        return a.copy()

    delta = a - b
    mag = np.abs(delta)
    lift = (mag > 0) & (mag < floor)
    out = delta.copy()
    out[lift] = np.sign(delta[lift]) * floor
    return (b + out).astype(np.float32)


def aggregate_retention(
    results: Iterable[TensorSurvival], *, field: str = "directional_permanence"
) -> float:
    """Pool a per-weight survival field across tensors.

    Weighted by the number of edited weights, so a tensor carrying most of the
    repair counts for most of the score, and untouched tensors are skipped
    entirely rather than counted as perfect.
    """
    items = [r for r in results if r.n_edited > 0]
    total = sum(r.n_edited for r in items)
    if total == 0:
        return 1.0
    return sum(getattr(r, field) * r.n_edited for r in items) / total


def read_gguf_tensors(path: str | Path) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield ``(name, qtype_name, dequantized_f32)`` for each tensor in a GGUF file.

    Dequantization is implemented for every GGML type, including k-quants, so
    this works on any file ``llama-quantize`` produces.
    """
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(str(path), "r")
    for t in reader.tensors:
        qt = GGMLQuantizationType(t.tensor_type)
        raw = np.asarray(t.data)
        if qt in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            vals = raw.astype(np.float32)
        else:
            vals = dequantize(raw, qt).astype(np.float32)
        # GGUF stores dimensions reversed relative to the torch layout.
        shape = tuple(int(d) for d in reversed(t.shape))
        yield str(t.name), qt.name, vals.reshape(shape)


def from_gguf_pair(
    before_gguf: str | Path,
    after_gguf: str | Path,
    reference_before: dict[str, np.ndarray] | None = None,
    reference_after: dict[str, np.ndarray] | None = None,
) -> list[TensorSurvival]:
    """Measure survival from two real GGUF files.

    ``before_gguf`` and ``after_gguf`` are the quantized drifted and repaired
    models. The intended edit is taken from the full-precision references when
    given; otherwise it falls back to the difference between the two
    dequantized files, which conflates the edit with quantization noise and is
    only a rough screen.
    """
    before = {n: v for n, _, v in read_gguf_tensors(before_gguf)}
    qtypes: dict[str, str] = {}
    after: dict[str, np.ndarray] = {}
    for n, qt, v in read_gguf_tensors(after_gguf):
        after[n] = v
        qtypes[n] = qt

    out: list[TensorSurvival] = []
    for name in sorted(set(before) & set(after)):
        b_q, a_q = before[name], after[name]
        if b_q.shape != a_q.shape:
            continue
        if reference_before is not None and reference_after is not None:
            if name not in reference_before or name not in reference_after:
                continue
            intended = reference_after[name].astype(np.float64) - reference_before[name].astype(
                np.float64
            )
        else:
            intended = a_q.astype(np.float64) - b_q.astype(np.float64)
        realized = a_q.astype(np.float64) - b_q.astype(np.float64)
        out.append(
            compare_edits(
                intended, realized, name=name, qtype=qtypes[name], n_params=b_q.size
            )
        )
    return out
