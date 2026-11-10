"""Per-tensor survival of a repair edit across one quantization, from real GGUF files.

For every tensor of a (drifted, repaired) pair:

    intended = repaired_BF16 - drifted_BF16      what the repair asked for, exact
    realized = deq(repaired_Q) - deq(drifted_Q)  what the deployed files express

and :func:`survival.compare_edits` turns the pair into per-weight permanence,
directional permanence and the diagnostics. The BF16 files are the lossless
conversions of the checkpoints, so ``intended`` contains no quantization noise.

Files are memory-mapped and processed one tensor at a time, so peak memory is a
few copies of the largest tensor (the 262,144 x 2,048 tied embedding), not of
the model. The empirical bin width of each tensor at this precision is recorded
too, measured as reconstruction error of the drifted weights. That is the
quantity the magnitude floor (MANSU Eq. 4) needs for k-quants, which NumPy
cannot quantize.

Runs in the CPU llama.cpp image (``gguf`` + ``numpy``).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .survival import compare_edits

_LAYER = re.compile(r"^blk\.(\d+)\.")


def _index(path: str | Path) -> dict[str, Any]:
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(str(path), "r")
    return {str(t.name): t for t in reader.tensors}


def _deq(t: Any) -> np.ndarray:
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    qt = GGMLQuantizationType(t.tensor_type)
    raw = np.asarray(t.data)
    if qt == GGMLQuantizationType.F32:
        vals = raw.astype(np.float32, copy=False)
    elif qt == GGMLQuantizationType.F16:
        vals = raw.astype(np.float32)
    else:
        vals = dequantize(raw, qt).astype(np.float32, copy=False)
    return vals.reshape(tuple(int(d) for d in reversed(t.shape)))


def module_of(name: str) -> str:
    """GGUF tensor name to a coarse module kind (``attn_q``, ``ffn_down``, ``token_embd``...)."""
    base = _LAYER.sub("", name)
    return base.rsplit(".", 1)[0] if base.endswith((".weight", ".bias")) else base


def survival_map(drift_bf16: str, repair_bf16: str, drift_q: str, repair_q: str,
                 *, qtype_label: str) -> dict[str, Any]:
    from gguf.constants import GGMLQuantizationType

    t0 = time.time()
    db, rb, dq, rq = (_index(p) for p in (drift_bf16, repair_bf16, drift_q, repair_q))
    names = sorted(set(db) & set(rb) & set(dq) & set(rq))
    rows: list[dict[str, Any]] = []
    for name in names:
        before, after = _deq(db[name]), _deq(rb[name])
        if before.shape != after.shape:
            continue
        intended = after.astype(np.float64) - before.astype(np.float64)
        if not np.any(intended):
            # Untouched by the repair: nothing to survive. Recorded so tensor
            # counts add up, skipped by every aggregate.
            rows.append({"name": name, "module": module_of(name), "n_edited": 0,
                         "n_params": int(before.size)})
            continue
        bq, aq = _deq(dq[name]), _deq(rq[name])
        realized = aq.astype(np.float64) - bq.astype(np.float64)
        s = compare_edits(intended, realized, name=name,
                          qtype=GGMLQuantizationType(rq[name].tensor_type).name,
                          n_params=int(before.size))
        err = np.abs(bq.astype(np.float64) - before.astype(np.float64))
        nz = err[err > 0]
        m = _LAYER.match(name)
        rows.append({
            **s.as_dict(),
            "module": module_of(name),
            "layer": int(m.group(1)) if m else None,
            "bin_width_est": float(np.median(nz) * 4.0) if nz.size else 0.0,
            "edit_to_bin_ratio_median": (float(s.median_abs_intended / (np.median(nz) * 4.0))
                                         if nz.size else None),
        })
    edited = [r for r in rows if r.get("n_edited", 0) > 0]
    total = sum(r["n_edited"] for r in edited)

    def pooled(field: str, subset: list[dict]) -> float | None:
        w = sum(r["n_edited"] for r in subset)
        return (sum(r[field] * r["n_edited"] for r in subset) / w) if w else None

    by_module: dict[str, dict] = {}
    for mod in sorted({r["module"] for r in edited}):
        sub = [r for r in edited if r["module"] == mod]
        by_module[mod] = {
            "n_tensors": len(sub), "n_edited": sum(r["n_edited"] for r in sub),
            "permanence": pooled("permanence", sub),
            "directional_permanence": pooled("directional_permanence", sub),
        }
    return {
        "qtype": qtype_label,
        "n_tensors": len(names),
        "n_tensors_edited": len(edited),
        "n_weights_edited": total,
        "share_of_weights_edited": total / max(1, sum(r["n_params"] for r in rows)),
        "permanence": pooled("permanence", edited),
        "directional_permanence": pooled("directional_permanence", edited),
        "erased_fraction": pooled("erased_fraction", edited),
        "aligned_retention_diagnostic": pooled("aligned_retention", edited),
        "by_module": by_module,
        "tensors": rows,
        "seconds": round(time.time() - t0, 1),
    }
