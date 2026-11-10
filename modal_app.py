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

from pathlib import Path

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


def _hf_token() -> str:
    """Read HF_TOKEN from the environment, falling back to the project ``.env``.

    Deliberately self-contained: this runs at module import time, which happens
    both locally (where it must find the token to ship) and inside the container
    (where the project package is not installed). Importing project code here
    would crash the container at startup. Returns "" rather than raising, so a
    missing token fails at call time with a clear message instead of breaking
    module import.
    """
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith("HF_TOKEN="):
                    return line.partition("=")[2].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


# Built once at import. Modal captures the value locally and injects it as an
# environment variable in the container.
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token()})


MODEL_ID = "CohereLabs/tiny-aya-fire"
BASE_ID = "CohereLabs/tiny-aya-base"
OFFICIAL_GGUF_REPO = "CohereLabs/tiny-aya-fire-GGUF"

# Each of our quantizations is checked against Cohere's published file of the same type.
# Q8_0 is the converter check: llama.cpp's Q8_0 path takes no importance matrix, so a
# mismatch there points at conversion, not at quantization settings. Q4_K_M is the file
# most people actually download.
VERIFY_QTYPES = {
    "Q8_0": "tiny-aya-fire-q8_0.gguf",
    "Q4_K_M": "tiny-aya-fire-q4_k_m.gguf",
}


def _download_access_problem(repo_id: str, token: str) -> str | None:
    """Return None if a real file of ``repo_id`` downloads, otherwise how to fix it.

    Requests the headers of an actual file, which fails exactly when a download
    would. ``model_info()`` is not a valid access check: it returns metadata,
    file list included, for gated repos the account has no access to.
    """
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import GatedRepoError

    try:
        get_hf_file_metadata(hf_hub_url(repo_id, "config.json"), token=token or None)
    except GatedRepoError:
        return (
            f"No download access to {repo_id}. Log in to huggingface.co as the token's owner, "
            f"open https://huggingface.co/{repo_id} and click 'Agree and access repository'. "
            "The repo is auto-approved, so access starts immediately."
        )
    return None


@app.function(
    image=llamacpp_image,
    volumes={VOL: volume},
    secrets=[hf_secret],
    cpu=8.0,
    memory=32768,
    timeout=60 * 90,
)
def build_and_verify_gguf(force: bool = False) -> dict:
    """Convert Tiny Aya Fire to GGUF ourselves, then check it against Cohere's official files.

    This is the project's external-validity anchor. Every survival number is
    measured on GGUF files this pipeline produces, so if those files match the
    artifacts Cohere actually ships, the measurements describe the real
    deployment path rather than a private approximation.

    CPU only: conversion and quantization do not use the GPU.
    """
    import hashlib
    import json
    import os
    import subprocess
    import time
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader
    from gguf.quants import dequantize
    from huggingface_hub import hf_hub_download, snapshot_download

    work = Path(VOL) / "day1"
    work.mkdir(parents=True, exist_ok=True)
    bin_dir = Path("/opt/llama.cpp/build/bin")
    report: dict = {"steps": []}

    def step(name: str, **kw):
        report["steps"].append({"step": name, **kw})

    def finish() -> dict:
        (work / "report.json").write_text(json.dumps(report, indent=2, default=str))
        volume.commit()
        return report

    problem = _download_access_problem(MODEL_ID, os.environ.get("HF_TOKEN", ""))
    if problem:
        step("access_check", ok=False, error=problem)
        return report

    # 1. Source weights. Always called: snapshot_download skips files already
    # present, and a crash mid-download must not leave a half-populated folder
    # that a cheaper "config.json exists" check would mistake for complete.
    t0 = time.time()
    src = work / "tiny-aya-fire"
    snapshot_download(MODEL_ID, local_dir=str(src),
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    cfg = json.loads((src / "config.json").read_text())
    step("download_model", seconds=round(time.time() - t0, 1),
         architectures=cfg.get("architectures"),
         num_hidden_layers=cfg.get("num_hidden_layers"),
         hidden_size=cfg.get("hidden_size"),
         vocab_size=cfg.get("vocab_size"),
         tie_word_embeddings=cfg.get("tie_word_embeddings"),
         model_type=cfg.get("model_type"),
         files=sorted(p.name for p in src.iterdir() if p.is_file()))

    # 2. Our own lossless conversion.
    t0 = time.time()
    ours_bf16 = work / "ours-bf16.gguf"
    if force or not ours_bf16.exists():
        tmp = ours_bf16.with_suffix(".partial")
        r = subprocess.run(
            ["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(src),
             "--outfile", str(tmp), "--outtype", "bf16"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            step("convert", ok=False, stderr=(r.stderr or "")[-4000:])
            return finish()
        tmp.rename(ours_bf16)
    step("convert", ok=True, seconds=round(time.time() - t0, 1),
         size_gb=round(ours_bf16.stat().st_size / 1e9, 2))
    llama_commit = subprocess.run(["git", "-C", "/opt/llama.cpp", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    report["llama_cpp_commit"] = llama_commit

    def field_value(field) -> object:
        try:
            return field.contents()
        except Exception as e:  # noqa: BLE001 - older gguf-py lacks contents()
            return f"unreadable ({type(e).__name__})"

    def describe(path: Path) -> tuple[dict, dict]:
        """Tensor index plus the metadata that says how a file was made."""
        reader = GGUFReader(str(path), "r")
        tensors = {
            str(t.name): (GGMLQuantizationType(t.tensor_type).name,
                          tuple(int(d) for d in reversed(t.shape)), t)
            for t in reader.tensors
        }
        meta: dict = {}
        for key, field in reader.fields.items():
            if key.startswith("quantize.") or key in ("general.file_type",
                                                      "general.quantization_version"):
                meta[key] = field_value(field)
            elif key == "tokenizer.chat_template":
                tpl = str(field_value(field))
                meta["chat_template_sha256"] = hashlib.sha256(tpl.encode()).hexdigest()[:16]
                meta["chat_template_chars"] = len(tpl)
        meta["used_imatrix"] = any(k.startswith("quantize.imatrix") for k in meta)
        return tensors, meta

    def compare(ours: Path, official: Path) -> dict:
        mine, ours_meta = describe(ours)
        theirs, official_meta = describe(official)
        common = sorted(set(mine) & set(theirs))
        same_type = [n for n in common if mine[n][0] == theirs[n][0] and mine[n][1] == theirs[n][1]]
        # Raw-byte equality is exact and cheap on memory-mapped files.
        identical = [n for n in same_type
                     if np.array_equal(np.asarray(mine[n][2].data), np.asarray(theirs[n][2].data))]
        differing = [n for n in same_type if n not in set(identical)]
        # Size the differences on the largest differing tensors.
        numeric = []
        for n in sorted(differing, key=lambda k: -int(np.prod(mine[k][1])))[:6]:
            qt = GGMLQuantizationType[mine[n][0]]
            a = dequantize(np.asarray(mine[n][2].data), qt).astype(np.float32)
            b = dequantize(np.asarray(theirs[n][2].data), qt).astype(np.float32)
            denom = float(np.abs(b).mean()) or 1.0
            numeric.append({"tensor": n, "qtype": mine[n][0],
                            "share_of_values_equal": float((a == b).mean()),
                            "max_abs_diff": float(np.abs(a - b).max()),
                            "mean_abs_diff_rel": float(np.abs(a - b).mean() / denom)})
        return {
            "n_tensors_ours": len(mine),
            "n_tensors_official": len(theirs),
            "only_in_ours": sorted(set(mine) - set(theirs))[:20],
            "only_in_official": sorted(set(theirs) - set(mine))[:20],
            "type_distribution_ours": dict(Counter(v[0] for v in mine.values())),
            "type_distribution_official": dict(Counter(v[0] for v in theirs.values())),
            "n_same_type_and_shape": len(same_type),
            "n_bit_identical": len(identical),
            "type_mismatches_sample": [
                {"tensor": n, "ours": mine[n][0], "official": theirs[n][0]}
                for n in common if mine[n][0] != theirs[n][0]
            ][:20],
            "differing_sample": numeric,
            "metadata_ours": ours_meta,
            "metadata_official": official_meta,
            "bit_identical_file_tensors": len(identical) == len(mine) == len(theirs),
        }

    # 3-5. For each verified type: quantize with llama.cpp defaults (as a normal
    # user would), fetch Cohere's file, compare tensor by tensor.
    report["comparisons"] = {}
    for qtype, official_name in VERIFY_QTYPES.items():
        t0 = time.time()
        ours_q = work / f"ours-{qtype.lower()}.gguf"
        if force or not ours_q.exists():
            tmp = ours_q.with_suffix(".partial")
            r = subprocess.run([str(bin_dir / "llama-quantize"), str(ours_bf16), str(tmp),
                                qtype, "8"], capture_output=True, text=True)
            if r.returncode != 0:
                step(f"quantize_{qtype}", ok=False, stderr=(r.stderr or "")[-4000:])
                return finish()
            tmp.rename(ours_q)
        step(f"quantize_{qtype}", ok=True, seconds=round(time.time() - t0, 1),
             size_gb=round(ours_q.stat().st_size / 1e9, 3))

        t0 = time.time()
        official = Path(hf_hub_download(OFFICIAL_GGUF_REPO, official_name,
                                        local_dir=str(work / "official")))
        step(f"download_official_{qtype}", seconds=round(time.time() - t0, 1),
             size_gb=round(official.stat().st_size / 1e9, 3))

        t0 = time.time()
        report["comparisons"][qtype] = compare(ours_q, official)
        step(f"compare_{qtype}", seconds=round(time.time() - t0, 1))

    return finish()


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


@app.local_entrypoint()
def day1(force: bool = False):
    """Verify our GGUF pipeline against Cohere's official quantization."""
    import json

    report = build_and_verify_gguf.remote(force=force)
    text = json.dumps(report, indent=2, default=str)
    out = Path(__file__).parent / "results" / "day1_gguf_verification.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {out}")
