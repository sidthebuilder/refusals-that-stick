"""Do weight-space safety repairs survive deployment quantization?

Lexsi AI Safety and Alignment Hackathon, Track 2b.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .survival import (
    TensorSurvival,
    aggregate_retention,
    apply_magnitude_floor,
    block_size,
    can_roundtrip,
    compare_edits,
    effective_bin_width,
    from_gguf_pair,
    quant_roundtrip,
    read_gguf_tensors,
    resolve_qtype,
    tensor_survival,
)

__all__ = [
    "__version__",
    "TensorSurvival",
    "aggregate_retention",
    "apply_magnitude_floor",
    "block_size",
    "can_roundtrip",
    "compare_edits",
    "effective_bin_width",
    "from_gguf_pair",
    "quant_roundtrip",
    "read_gguf_tensors",
    "resolve_qtype",
    "tensor_survival",
]
