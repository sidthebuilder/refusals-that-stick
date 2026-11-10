"""Do weight-space safety repairs survive deployment quantization?

Lexsi AI Safety and Alignment Hackathon, Track 2b.

Submodules are imported lazily. Each Modal image installs only what its stage
needs (the data image has no ``gguf``, the llama.cpp image has no ``datasets``),
so importing the package must not pull in every dependency.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_SURVIVAL = (
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
)

__all__ = ["__version__", *_SURVIVAL]


def __getattr__(name: str) -> Any:
    if name in _SURVIVAL:
        return getattr(importlib.import_module(".survival", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
