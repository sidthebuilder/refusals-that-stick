"""Build the GGUF precision ladder for one checkpoint with the pinned llama.cpp.

Conversion is lossless (BF16); every lower precision is quantized from that one
file with ``llama-quantize`` defaults, the way a developer publishing a model
would. The fixes reuse the same path with extra arguments: ``--tensor-type`` for
circuit-guided precision, ``--imatrix`` for importance matrices.

Before quantizing, the BF16 file's chat template is checked against Fire's
(hash recorded on day 1). A drifted or repaired checkpoint whose GGUF were
prompted differently from Fire would confound every comparison.

Runs in the CPU llama.cpp image.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Sequence

LLAMA = Path("/opt/llama.cpp")
PRECISIONS = ("BF16", "Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K")
# sha256 prefix of tokenizer.chat_template in Fire's official GGUFs (day 1).
FIRE_TEMPLATE_SHA = "2147c474f41b522b"


def chat_template_sha(gguf: str | Path) -> str | None:
    from gguf.gguf_reader import GGUFReader

    field = GGUFReader(str(gguf), "r").fields.get("tokenizer.chat_template")
    if field is None:
        return None
    return hashlib.sha256(str(field.contents()).encode()).hexdigest()[:16]


def convert(hf_dir: str | Path, out: str | Path) -> dict:
    out = Path(out)
    tmp = out.with_suffix(".partial")
    t0 = time.time()
    r = subprocess.run(["python", str(LLAMA / "convert_hf_to_gguf.py"), str(hf_dir),
                        "--outfile", str(tmp), "--outtype", "bf16"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"convert failed:\n{(r.stderr or '')[-3000:]}")
    tmp.rename(out)
    return {"seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size}


def quantize(src: str | Path, out: str | Path, qtype: str, *, threads: int = 8,
             extra_args: Sequence[str] = ()) -> dict:
    out = Path(out)
    tmp = out.with_suffix(".partial")
    t0 = time.time()
    cmd = [str(LLAMA / "build/bin/llama-quantize"), *extra_args, str(src), str(tmp), qtype,
           str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"quantize {qtype} failed:\n{(r.stderr or r.stdout or '')[-3000:]}")
    tmp.rename(out)
    return {"seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size,
            "args": list(extra_args)}


def build_ladder(hf_dir: str | Path, out_dir: str | Path,
                 precisions: Sequence[str] = PRECISIONS, *, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"hf_dir": str(hf_dir), "files": {}}
    bf16 = out_dir / "BF16.gguf"
    if force or not bf16.exists():
        report["convert"] = convert(hf_dir, bf16)
    sha = chat_template_sha(bf16)
    report["chat_template_sha"] = sha
    if sha != FIRE_TEMPLATE_SHA:
        raise RuntimeError(f"{bf16} chat template {sha} differs from Fire's "
                           f"{FIRE_TEMPLATE_SHA}; refusing to build a confounded ladder")
    report["files"]["BF16"] = {"bytes": bf16.stat().st_size}
    for p in precisions:
        if p == "BF16":
            continue
        target = out_dir / f"{p}.gguf"
        if force or not target.exists():
            report["files"][p] = quantize(bf16, target, p)
        else:
            report["files"][p] = {"bytes": target.stat().st_size, "cached": True}
    return report
