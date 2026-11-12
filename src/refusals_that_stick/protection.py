"""Mixed-precision protection of a repair edit, computed from per-tensor survival maps.

Fix 1 of the pre-registration keeps chosen tensors at Q8_0 inside a lower-precision
GGUF file (``llama-quantize --tensor-type``). Q8_0 quantizes each tensor on its
own, so a protected tensor is bit-identical to the same tensor in the full Q8_0
file, and an unprotected tensor is identical to its version in the base file.
Given the per-tensor survival maps measured at Q8_0 and at the base precision,
the weight-level result of any protection set is therefore exact arithmetic:

    each edited tensor contributes its Q8_0 row if protected, its base row if not,

pooled by the number of edited weights, as in :mod:`survival_map`. File-size cost
is computed from each tensor's parameter count and GGML type.

This rests on one assumption, that ``--tensor-type`` leaves every other tensor's
type unchanged. It is checked by building one protected file and measuring it.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

__all__ = ["BITS_PER_WEIGHT", "CIRCUIT_MODULES", "circuit_tensor_names", "pooled_survival",
           "protected_survival", "size_bytes", "random_matched_set"]

# Bits per weight of each GGML type: block bytes x 8 / block size.
BITS_PER_WEIGHT = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 34 * 8 / 32,       # 8.5
    "Q6_K": 210 * 8 / 256,     # 6.5625
    "Q5_K": 176 * 8 / 256,     # 5.5
    "Q4_K": 144 * 8 / 256,     # 4.5
    "Q4_0": 18 * 8 / 32,       # 4.5
    "Q3_K": 110 * 8 / 256,     # 3.4375
    "Q2_K": 84 * 8 / 256,      # 2.625
}

# Hugging Face projection names (as CircuitKIT reports them) to GGUF tensor names.
CIRCUIT_MODULES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v",
                   "o_proj": "attn_output", "gate_proj": "ffn_gate", "up_proj": "ffn_up",
                   "down_proj": "ffn_down"}


def circuit_tensor_names(layers: Iterable[int], modules: Iterable[str]) -> list[str]:
    """GGUF weight names for every (layer, projection) pair of a circuit."""
    out = []
    for layer in sorted(set(layers)):
        for m in modules:
            if m not in CIRCUIT_MODULES:
                raise ValueError(f"unknown projection {m!r}")
            out.append(f"blk.{layer}.{CIRCUIT_MODULES[m]}.weight")
    return sorted(set(out))


def pooled_survival(rows: Sequence[dict]) -> dict[str, float | int | None]:
    """Edit-weighted survival over per-tensor rows, as survival_map pools it."""
    edited = [r for r in rows if r.get("n_edited", 0) > 0]
    w = sum(r["n_edited"] for r in edited)
    if w == 0:
        return {"n_edited": 0, "permanence": None, "directional_permanence": None,
                "erased_fraction": None, "direction_among_moved": None}

    def avg(field: str) -> float:
        return sum(r[field] * r["n_edited"] for r in edited) / w

    perm, dirp = avg("permanence"), avg("directional_permanence")
    return {"n_edited": w, "permanence": perm, "directional_permanence": dirp,
            "erased_fraction": avg("erased_fraction"),
            "direction_among_moved": dirp / perm if perm > 0 else None}


def protected_survival(base_rows: Sequence[dict], q8_rows: Sequence[dict],
                       protect: Iterable[str]) -> dict[str, Any]:
    """Survival of the edit when ``protect`` is kept at Q8_0 and the rest at the base type."""
    q8 = {r["name"]: r for r in q8_rows}
    protect = set(protect)
    mixed, n_protected_edited = [], 0
    for r in base_rows:
        if r.get("n_edited", 0) > 0 and r["name"] in protect:
            if r["name"] not in q8:
                raise KeyError(f"{r['name']} missing from the Q8_0 map")
            mixed.append(q8[r["name"]])
            n_protected_edited += 1
        else:
            mixed.append(r)
    out = pooled_survival(mixed)
    out["n_tensors_protected"] = len(protect)
    out["n_edited_tensors_protected"] = n_protected_edited
    return out


def size_bytes(type_rows: Sequence[dict], protect: Iterable[str] = ()) -> float:
    """File payload in bytes: each tensor at its own type, protected tensors at Q8_0.

    ``type_rows`` needs ``name``, ``n_params`` and ``qtype`` for every tensor. F32
    tensors (norms) are never re-typed. Metadata is not included.
    """
    protect = set(protect)
    total = 0.0
    for r in type_rows:
        qt = r["qtype"]
        if r["name"] in protect and qt != "F32":
            qt = "Q8_0"
        total += r["n_params"] * BITS_PER_WEIGHT[qt] / 8
    return total


def random_matched_set(type_rows: Sequence[dict], target_params: int, seed: int,
                       tolerance: float = 0.02) -> list[str]:
    """Random quantized tensors whose parameter count is within ``tolerance`` of the target.

    Equal parameters means equal bytes at Q8_0, which is the pre-registered match.
    Tensors that would overshoot the upper bound are skipped rather than added.
    """
    pool = [r for r in type_rows if r["qtype"] != "F32"]
    rng = random.Random(seed)
    rng.shuffle(pool)
    hi = target_params * (1 + tolerance)
    lo = target_params * (1 - tolerance)
    chosen, total = [], 0
    for r in pool:
        if total >= lo:
            break
        if total + r["n_params"] <= hi:
            chosen.append(r["name"])
            total += r["n_params"]
    if total < lo:
        raise RuntimeError(f"seed {seed}: could not reach {target_params} parameters")
    return sorted(chosen)
