"""Modal environments for the four Lexsi toolkits, plus llama.cpp.

Why separate images: the toolkits do not share a dependency set. SafeTune
requires Python >= 3.12 (`pyproject.toml`), CuratorKIT >= 3.11, AuditKIT and
CircuitKIT >= 3.10. CircuitKIT additionally pins transformer-lens 3.8 with
transformers 5, which is not a combination the others ask for. Forcing one
image would mean resolving conflicts that do not need resolving.

Everything here is CPU-only and cheap. The point of this file on day zero is to
prove the images build and the imports work *before* any GPU time is bought.

Usage:
    modal run modal_app.py::smoke_all
"""

from __future__ import annotations

import modal

app = modal.App("refusals-that-stick")

# Shared storage for checkpoints, GGUF files and results. Named so it survives
# between runs; the grid is resumable rather than restarted on failure.
volume = modal.Volume.from_name("rts-artifacts", create_if_missing=True)
VOL = "/vol"

_TORCH = ["torch==2.9.0", "transformers>=5.0", "accelerate", "safetensors", "peft", "datasets"]

# SafeTune: the repair methods (C-ΔΘ, RESTA, Q-resafe). Needs 3.12+.
safetune_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_TORCH)
    .pip_install("safetune")
)

# CircuitKIT: refusal-circuit discovery. Pins transformer-lens.
circuitkit_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_TORCH)
    .pip_install("circuitkit")
)

# AuditKIT + CuratorKIT: evaluation and data curation.
eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_TORCH)
    .pip_install("auditkit[transformers]", "curatorkit")
)

# llama.cpp: converts to GGUF and quantizes. This is the deployment path whose
# effect on repairs the project measures, so it must be the real binary rather
# than a reimplementation.
llamacpp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF",
        "cmake --build /opt/llama.cpp/build -j --target llama-quantize llama-cli llama-imatrix",
    )
    .pip_install("numpy", "gguf", "torch==2.9.0", "transformers>=5.0", "sentencepiece", "protobuf")
)


def _report(label: str, probes: dict[str, str]) -> dict:
    """Import each probe and report version or error, without raising."""
    import importlib

    found = {}
    for name, attr in probes.items():
        try:
            mod = importlib.import_module(name)
            found[name] = str(getattr(mod, attr, "imported"))
        except Exception as e:  # noqa: BLE001 - we want the failure text, not a crash
            found[name] = f"FAILED {type(e).__name__}: {e}"
    import sys

    return {"image": label, "python": sys.version.split()[0], "modules": found}


@app.function(image=safetune_image, volumes={VOL: volume}, timeout=900)
def smoke_safetune() -> dict:
    out = _report("safetune", {"torch": "__version__", "transformers": "__version__",
                               "safetune": "__version__"})
    try:
        from safetune.interventions.recover.ctheta import apply_ctheta  # noqa: F401

        out["apply_ctheta"] = "importable"
    except Exception as e:  # noqa: BLE001
        out["apply_ctheta"] = f"FAILED {type(e).__name__}: {e}"
    return out


@app.function(image=circuitkit_image, volumes={VOL: volume}, timeout=900)
def smoke_circuitkit() -> dict:
    out = _report("circuitkit", {"torch": "__version__", "transformers": "__version__",
                                 "circuitkit": "__version__"})
    try:
        import circuitkit as ck

        out["has_discover"] = hasattr(ck, "discover")
        out["has_quantize"] = hasattr(ck, "quantize")
        # cohere2 is Tiny Aya's architecture; confirm the registry knows it.
        try:
            from circuitkit.models import architecture_registry as reg  # type: ignore

            out["registry"] = str(getattr(reg, "__name__", "present"))
        except Exception as e:  # noqa: BLE001
            out["registry"] = f"not found: {type(e).__name__}"
    except Exception as e:  # noqa: BLE001
        out["circuitkit_detail"] = f"FAILED {type(e).__name__}: {e}"
    return out


@app.function(image=eval_image, volumes={VOL: volume}, timeout=900)
def smoke_eval() -> dict:
    return _report("eval", {"torch": "__version__", "transformers": "__version__",
                            "auditkit": "__version__", "curatorkit": "__version__"})


@app.function(image=llamacpp_image, volumes={VOL: volume}, timeout=900)
def smoke_llamacpp() -> dict:
    """Confirm the binaries exist and the conversion script is present."""
    import subprocess
    from pathlib import Path

    out: dict = {"image": "llamacpp"}
    build = Path("/opt/llama.cpp/build/bin")
    for tool in ("llama-quantize", "llama-cli", "llama-imatrix"):
        p = build / tool
        out[tool] = p.exists()
    conv = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
    out["convert_hf_to_gguf.py"] = conv.exists()

    # llama-quantize with no arguments prints its usage, which lists the quant
    # types we depend on. Confirm the ones in the pre-registered grid exist.
    try:
        r = subprocess.run([str(build / "llama-quantize"), "--help"],
                           capture_output=True, text=True, timeout=120)
        text = (r.stdout or "") + (r.stderr or "")
        out["quant_types_present"] = {
            q: (q in text) for q in ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K")
        }
        out["supports_tensor_type_flag"] = "--tensor-type" in text
        out["supports_imatrix_flag"] = "--imatrix" in text
    except Exception as e:  # noqa: BLE001
        out["usage_probe"] = f"FAILED {type(e).__name__}: {e}"
    return out


@app.local_entrypoint()
def smoke_all():
    """Build every image and verify imports and binaries. CPU only."""
    import json

    results = [
        smoke_safetune.remote(),
        smoke_circuitkit.remote(),
        smoke_eval.remote(),
        smoke_llamacpp.remote(),
    ]
    print(json.dumps(results, indent=2, default=str))
