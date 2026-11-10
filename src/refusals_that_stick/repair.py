"""Apply a SafeTune weight-space repair to the drifted model.

Primary methods (pre-registered), both adding the safety vector ``Fire - base``:

* **RESTA** (``apply_resta``): to every tensor;
* **C-ΔΘ** (``apply_ctheta``): only to the circuit's (layer, module) tensors.

Pre-registered fallback, used because neither primary method restored 50% of
the drift at BF16 without wrecking capability (SPRINT_LOG, 11 Nov):

* **LoX** (``apply_lox``, recover form): ``W + c * LowRank_k(Fire - base)`` on
  every attention and MLP matrix, so only the top-k singular directions of the
  alignment delta (the LoX "safety subspace") are re-applied, not the whole
  instruction-tuning delta that RESTA re-applies. The tied embedding is
  excluded with ``param_filter``; otherwise SafeTune would SVD the 262,144 x
  2,048 matrix twice, once under each name.
* **SafeMerge** (``apply_safemerge``): per weight matrix, cosine between the
  fine-tuning update ``W_drift - W_fire`` and its projection on the safety
  subspace of ``Fire - base``. Matrices below the threshold (paper default
  0.35) are interpolated back toward Fire, ``(1 - a) W_drift + a W_fire``.
  The rest are left as fine-tuned. So the edit is sparse by layer.

Models are loaded on CPU in bf16. The edited tensors are recorded by comparing
weights before and after, so each report states exactly how sparse the repair
is. Configs are restored from Fire afterwards, as for the drifted model.

Runs in the SafeTune image.
"""

from __future__ import annotations

import time
from typing import Any

from .drift import restore_config_files, safetensors_index

METHODS = ("resta", "ctheta", "lox", "safemerge")
SAFEMERGE_THRESHOLD = 0.35  # SafeMERGE paper / SafeTune default


def repair(*, method: str, drifted_dir: str, fire_dir: str, base_dir: str, out_dir: str,
           strength: float, circuit_info_path: str | None = None, dare: bool = False,
           seed: int = 0, rank: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    t0 = time.time()

    def load(path: str):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")
        m.eval()
        return m

    target, aligned, base = load(drifted_dir), load(fire_dir), load(base_dir)
    before = {k: v.detach().clone() for k, v in target.state_dict().items()
              if k.endswith(".weight")}
    extra: dict[str, Any] = {}
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
        elif method == "lox":
            from safetune.interventions.recover.lox import apply_lox

            if rank is None:
                raise ValueError("LoX needs rank")
            apply_lox(target, base=base, aligned=aligned, rank=int(rank),
                      extrapolation_factor=strength, param_filter=["self_attn", "mlp"])
            extra["rank"] = int(rank)
        elif method == "safemerge":
            from safetune.interventions.recover.safemerge import apply_safemerge

            apply_safemerge(target, base=base, aligned=aligned,
                            threshold=SAFEMERGE_THRESHOLD, alpha=strength)
            extra["threshold"] = SAFEMERGE_THRESHOLD

    after = target.state_dict()
    edited = sorted(k for k, v in before.items() if not torch.equal(v, after[k]))
    total = sum(v.numel() for v in before.values())
    edited_params = sum(before[k].numel() for k in edited)
    layers = sorted({int(p) for k in edited for p in k.split(".")[2:3] if p.isdigit()})
    del aligned, base, before
    target.save_pretrained(out_dir)
    restored = restore_config_files(fire_dir, out_dir)
    if safetensors_index(out_dir) != safetensors_index(fire_dir):
        raise RuntimeError("repaired checkpoint does not match Fire's tensor layout")
    return {
        "method": method, "strength": strength, "dare": dare, "out_dir": out_dir,
        "circuit_info": circuit_info_path, **extra,
        "n_tensors_edited": len(edited),
        "share_of_weights_edited": edited_params / total if total else None,
        "layers_edited": layers,
        "edited_sample": edited[:12],
        "config_files": restored,
        "seconds": round(time.time() - t0, 1),
    }
