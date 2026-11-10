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

import json
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
#
# Pinned to the commit whose Q8_0 and Q4_K_M output was verified bit-identical
# to Cohere's published files (results/day1_gguf_verification.json). Unpinned,
# an image rebuild would silently pull a newer llama.cpp.
LLAMA_CPP_COMMIT = "90e3841"
_LLAMA_CLONE = (
    "git clone --filter=blob:none https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
    f"git -C /opt/llama.cpp checkout --quiet {LLAMA_CPP_COMMIT}",
)

llamacpp_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands(
        *_LLAMA_CLONE,
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF",
        "cmake --build /opt/llama.cpp/build -j --target llama-quantize llama-cli llama-imatrix",
    )
    .pip_install("numpy", "gguf", "torch==2.9.0", "transformers>=5.0", "sentencepiece", "protobuf")
)

# The same llama.cpp commit built for CUDA, for generation, perplexity and the
# importance matrix. Compute capability 8.9 covers the L40S and L4.
#
# Static libraries: the image builder has no GPU driver, so a shared
# libggml-cuda.so cannot resolve libcuda.so.1 when the executables are linked.
# Statically linked binaries link against the toolkit's driver stub instead and
# pick up the real driver at run time.
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libgomp1")
    .run_commands(
        *_LLAMA_CLONE,
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF "
        "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_BUILD_TESTS=OFF",
        "cmake --build /opt/llama.cpp/build -j --target llama-server llama-perplexity llama-imatrix",
    )
    .pip_install("numpy", "gguf", "transformers>=5.0", "tokenizers", "jinja2", "huggingface_hub",
                 "requests")
)

# The project package, mounted into every image that runs project code. Added at
# container start (not baked in), so editing it never triggers an image rebuild.
_SRC_PKG = Path(__file__).parent / "src" / "refusals_that_stick"


def _with_src(image: modal.Image) -> modal.Image:
    return image.add_local_dir(_SRC_PKG, remote_path="/root/refusals_that_stick",
                               ignore=["__pycache__", "*.pyc"])


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


# --------------------------------------------------------------------------- #
# Day 2: data, models, drift                                                   #
# --------------------------------------------------------------------------- #

MODELS = f"{VOL}/models"
FIRE_DIR = f"{MODELS}/fire"
BASE_DIR = f"{MODELS}/base"

data_image = _with_src(eval_image)
train_image = _with_src(safetune_image)


@app.function(image=data_image, volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0,
              memory=16384, timeout=60 * 40)
def build_data() -> dict:
    """Build every dataset from pinned public sources; see build_data.py."""
    import sys

    sys.path.insert(0, "/root")
    from refusals_that_stick.build_data import build_all

    out = build_all(VOL)
    volume.commit()
    return out


@app.function(image=eval_image, volumes={VOL: volume}, secrets=[hf_secret], cpu=4.0,
              memory=8192, timeout=60 * 30)
def fetch_models() -> dict:
    """Download Fire and its base model to the volume, pinned to the current revisions."""
    import os
    import time

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    out = {}
    for repo, local in ((MODEL_ID, FIRE_DIR), (BASE_ID, BASE_DIR)):
        problem = _download_access_problem(repo, os.environ.get("HF_TOKEN", ""))
        if problem:
            out[repo] = {"ok": False, "error": problem}
            continue
        t0 = time.time()
        sha = api.model_info(repo).sha
        snapshot_download(repo, revision=sha, local_dir=local,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                          "*.jinja"])
        files = sorted(p.name for p in Path(local).iterdir() if p.is_file())
        out[repo] = {"ok": True, "revision": sha, "dir": local, "files": files,
                     "seconds": round(time.time() - t0, 1)}
    (Path(MODELS) / "revisions.json").write_text(json.dumps(out, indent=2))
    volume.commit()
    return out


@app.function(image=train_image, gpu="A100-80GB", volumes={VOL: volume}, secrets=[hf_secret],
              memory=65536, timeout=60 * 90)
def train_drift_job(variant: str, dose_fraction: float, epochs: int = 3, lr: float = 2e-4,
                    batch_size: int = 4, max_len: int = 1024, seed: int = 0) -> dict:
    """LoRA fine-tune Fire on the curated Hindi set, optionally with a harmful dose."""
    import sys

    sys.path.insert(0, "/root")
    from refusals_that_stick.drift import train_drift

    volume.reload()  # see the data and models written by other containers
    out_dir = f"{MODELS}/drift-{variant}"
    result = train_drift(
        src_dir=FIRE_DIR,
        train_jsonl=f"{VOL}/data/train/hindi_sft_train.jsonl",
        dose_jsonl=f"{VOL}/private/harmful_dose.jsonl",
        dose_fraction=dose_fraction,
        out_dir=out_dir,
        adapter_dir=f"{MODELS}/drift-{variant}-adapter",
        epochs=epochs, lr=lr, batch_size=batch_size, max_len=max_len, seed=seed,
    )
    result["variant"] = variant
    (Path(out_dir) / "drift_report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=cuda_image, gpu="L40S", timeout=60 * 10)
def smoke_cuda() -> dict:
    """Confirm the CUDA build sees the GPU, before any evaluation depends on it."""
    import subprocess

    out: dict = {}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True)
    out["nvidia_smi"] = (smi.stdout or smi.stderr).strip()
    bin_dir = Path("/opt/llama.cpp/build/bin")
    out["binaries"] = {b: (bin_dir / b).exists()
                       for b in ("llama-server", "llama-perplexity", "llama-imatrix")}
    r = subprocess.run([str(bin_dir / "llama-server"), "--list-devices"],
                       capture_output=True, text=True, timeout=120)
    out["devices"] = ((r.stdout or "") + (r.stderr or "")).strip()[-1500:]
    out["commit"] = subprocess.run(["git", "-C", "/opt/llama.cpp", "rev-parse", "--short",
                                    "HEAD"], capture_output=True, text=True).stdout.strip()
    return out


# --------------------------------------------------------------------------- #
# GGUF ladder, judge, evaluation cells                                         #
# --------------------------------------------------------------------------- #

GGUF_ROOT = f"{VOL}/gguf"
CELLS = f"{VOL}/cells"
JUDGE_DIR = f"{VOL}/judge"
JUDGE_GGUF = f"{JUDGE_DIR}/qwen3guard-gen-8b-q8_0.gguf"
JUDGE_TOKENIZER = f"{JUDGE_DIR}/tokenizer"

quant_image = _with_src(llamacpp_image)
gpu_eval_image = _with_src(cuda_image)


@app.function(image=quant_image, volumes={VOL: volume}, cpu=8.0, memory=32768,
              timeout=60 * 60)
def build_ladder_job(model: str, precisions: list[str] | None = None, force: bool = False) -> dict:
    """BF16 conversion plus every pre-registered precision for one checkpoint."""
    import sys

    sys.path.insert(0, "/root")
    from refusals_that_stick.ladder import PRECISIONS, build_ladder

    volume.reload()
    report = build_ladder(f"{MODELS}/{model}", f"{GGUF_ROOT}/{model}",
                          precisions or list(PRECISIONS), force=force)
    report["model"] = model
    (Path(GGUF_ROOT) / model / "ladder_report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(image=quant_image, volumes={VOL: volume}, secrets=[hf_secret], cpu=8.0,
              memory=32768, timeout=60 * 60)
def convert_judge() -> dict:
    """Qwen3Guard-Gen-8B at a pinned revision, converted to Q8_0 by the pinned llama.cpp."""
    import shutil
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/root")
    from huggingface_hub import snapshot_download
    from refusals_that_stick.judge import JUDGE_REPO, JUDGE_REVISION

    if Path(JUDGE_GGUF).exists() and Path(JUDGE_TOKENIZER, "tokenizer_config.json").exists():
        return {"cached": True, "gguf": JUDGE_GGUF}
    t0 = time.time()
    hf_dir = Path(JUDGE_DIR) / "hf"
    snapshot_download(JUDGE_REPO, revision=JUDGE_REVISION, local_dir=str(hf_dir),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "merges.txt",
                                      "vocab.json"])
    tmp = Path(JUDGE_GGUF).with_suffix(".partial")
    r = subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(hf_dir),
                        "--outfile", str(tmp), "--outtype", "q8_0"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "stderr": (r.stderr or "")[-3000:]}
    tmp.rename(JUDGE_GGUF)
    tok_dir = Path(JUDGE_TOKENIZER)
    tok_dir.mkdir(parents=True, exist_ok=True)
    for f in hf_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copyfile(f, tok_dir / f.name)
    shutil.rmtree(hf_dir)  # 16 GB of weights no longer needed once converted
    volume.commit()
    return {"ok": True, "gguf": JUDGE_GGUF, "bytes": Path(JUDGE_GGUF).stat().st_size,
            "revision": JUDGE_REVISION, "seconds": round(time.time() - t0, 1)}


def _cell_tasks(split: str) -> list[str]:
    """Every task file present for a split (Hindi XSTest appears once translated)."""
    data = Path(VOL) / "data" / "eval"
    order = ["redteam_hi", "redteam_en", "xstest_en", "xstest_hi", "mmlu_en", "mmlu_hi"]
    return [f"{t}_{split}" for t in order if (data / f"{t}_{split}.jsonl").exists()]


@app.function(image=gpu_eval_image, gpu="L40S", volumes={VOL: volume}, secrets=[hf_secret],
              memory=32768, timeout=60 * 60, max_containers=10)
def eval_cell(model: str, precision: str, split: str = "dev", gguf: str | None = None) -> dict:
    """Generate, judge and measure perplexity for one (model, precision, split) cell."""
    import sys

    sys.path.insert(0, "/root")
    from refusals_that_stick.evaluate import run_cell

    volume.reload()
    gguf = gguf or f"{GGUF_ROOT}/{model}/{precision}.gguf"
    tasks = _cell_tasks(split)
    summary = run_cell(
        gguf=gguf,
        cell_dir=f"{CELLS}/{split}/{model}/{precision}",
        data_dir=f"{VOL}/data/eval",
        tasks=tasks,
        judge_gguf=JUDGE_GGUF,
        judge_tokenizer_dir=JUDGE_TOKENIZER,
        ppl_text=f"{VOL}/data/eval/hindi_heldout_ppl.txt",
    )
    summary.update(model=model, precision=precision, split=split)
    volume.commit()
    return summary


def _write_tree(base: Path, files: dict[str, str]) -> list[str]:
    written = []
    for rel, text in sorted(files.items()):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        written.append(rel)
    return written


@app.local_entrypoint()
def day2_setup():
    """Build the datasets and fetch both models in parallel; confirm the CUDA image."""
    root = Path(__file__).parent
    data_call = build_data.spawn()
    fetch_call = fetch_models.spawn()
    cuda_call = smoke_cuda.spawn()

    fetched = fetch_call.get()
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "model_revisions.json").write_text(json.dumps(fetched, indent=2))
    print("models:", json.dumps(fetched, indent=2))

    built = data_call.get()
    written = _write_tree(root / "data", built["files"])
    print(f"data: wrote {len(written)} files under data/")
    print(json.dumps(built["stats"], indent=2, ensure_ascii=False, default=str)[:6000])

    try:
        cuda = cuda_call.get()
    except Exception as e:  # noqa: BLE001 - report the build failure, keep the data
        cuda = {"error": f"{type(e).__name__}: {e}"}
    (root / "results" / "cuda_smoke.json").write_text(json.dumps(cuda, indent=2))
    print("cuda:", json.dumps(cuda, indent=2))


@app.local_entrypoint()
def fire_baseline(precisions: str = "BF16,Q4_K_M"):
    """Fire's GGUF ladder and judge, then the first dev cells as the integration test."""
    root = Path(__file__).parent
    judge_call = convert_judge.spawn()
    ladder = build_ladder_job.remote("fire")
    print("ladder:", json.dumps(ladder, indent=2))
    print("judge:", json.dumps(judge_call.get(), indent=2))
    cells = [("fire", p, "dev") for p in precisions.split(",")]
    summaries = []
    for s in eval_cell.starmap(cells, return_exceptions=True):
        summaries.append(s if isinstance(s, dict) else {"error": repr(s)})
        print(json.dumps(summaries[-1], indent=2, default=str)[:4000])
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "fire_baseline_dev.json").write_text(
        json.dumps({"ladder": ladder, "cells": summaries}, indent=2, default=str))


@app.local_entrypoint()
def drift(epochs: int = 3, lr: float = 2e-4):
    """Train the two drift variants in parallel; the pre-registered gate picks one later."""
    root = Path(__file__).parent
    calls = {name: train_drift_job.spawn(name, frac, epochs=epochs, lr=lr)
             for name, frac in (("plain", 0.0), ("dose3", 0.03))}
    reports = {}
    for name, call in calls.items():
        try:
            reports[name] = call.get()
        except Exception as e:  # noqa: BLE001 - one failed variant must not hide the other
            reports[name] = {"error": f"{type(e).__name__}: {e}"}
        print(name, json.dumps({k: v for k, v in reports[name].items() if k != "loss_curve"},
                               indent=2, default=str))
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "drift_training.json").write_text(json.dumps(reports, indent=2,
                                                                     default=str))


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
