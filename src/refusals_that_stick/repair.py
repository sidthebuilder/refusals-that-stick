"""Apply a SafeTune weight-space repair to the drifted model.

Both methods add the same safety vector, ``Fire - base`` (SafeTune's
``aligned - base``), scaled by a strength:

* **RESTA** (``apply_resta``): to every tensor;
* **C-ΔΘ** (``apply_ctheta``): only to the circuit's (layer, module) tensors,
  as listed by CircuitKIT's ``layer_suggestions``.

So the H0 contrast, sparse against dense, is exactly the restriction mask with
the vector held fixed. Models are loaded on CPU in bf16. RESTA streams one
tensor at a time and C-ΔΘ edits a subset, so 64 GB of host memory holds the
three 3.35B checkpoints. Configs are restored from Fire afterwards, as for the
drifted model.

Runs in the SafeTune image.
"""

from __future__ import annotations

import time
from typing import Any

from .drift import restore_config_files, safetensors_index


def repair(*, method: str, drifted_dir: str, fire_dir: str, base_dir: str, out_dir: str,
           strength: float, circuit_info_path: str | None = None, dare: bool = False,
           seed: int = 0) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()

    def load(path: str):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
        m.eval()
        return m

    target, aligned, base = load(drifted_dir), load(fire_dir), load(base_dir)
    before = {k: v.detach().clone() for k, v in target.state_dict().items()
              if k.endswith(".weight")} if method == "ctheta" else None
    with torch.no_grad():
        if method == "resta":
            from safetune.interventions.recover.resta import apply_resta

            apply_resta(target, base=base, aligned=aligned, alpha=strength, dare=dare,
                        dare_seed=seed if dare else None, device="cpu")
        elif method == "ctheta":
            from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
            from safetune.interventions.recover.ctheta import apply_ctheta

            if not circuit_info_path:
                raise ValueError("C-ΔΘ needs circuit_info_path")
            info = load_circuit_info_from_file(circuit_info_path)
            if info is None or info.layer_suggestions is None:
                raise ValueError(f"no layer_suggestions in {circuit_info_path}")
            apply_ctheta(target, aligned, base, info, strength=strength)
        else:
            raise ValueError(f"unknown method {method!r}")

    edited = None
    if before is not None:
        after = target.state_dict()
        edited = sorted(k for k, v in before.items() if not torch.equal(v, after[k]))
    del aligned, base
    target.save_pretrained(out_dir)
    restored = restore_config_files(fire_dir, out_dir)
    if safetensors_index(out_dir) != safetensors_index(fire_dir):
        raise RuntimeError("repaired checkpoint does not match Fire's tensor layout")
    return {
        "method": method, "strength": strength, "dare": dare, "out_dir": out_dir,
        "circuit_info": circuit_info_path,
        "n_tensors_edited": len(edited) if edited is not None else "all (RESTA)",
        "edited_sample": edited[:12] if edited else None,
        "config_files": restored,
        "seconds": round(time.time() - t0, 1),
    }
