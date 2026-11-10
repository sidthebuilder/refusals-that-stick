"""Cause safety drift the way it happens in practice: a plain LoRA fine-tune on Hindi data.

Uses SafeTune's undefended baseline, ``PlainSFTTrainer`` (LoRA r=16, alpha=32 on
every attention and MLP projection, loss on the assistant turn only). One
subclass changes two things and nothing about the training itself:

* the LoRA adapter is saved before merging, because the pre-registration
  releases the drifted model as an adapter only;
* the training loss history is kept for the report.

After saving, the tokenizer, chat template, generation config and model config
are restored byte-for-byte from the source model. ``tokenizer.save_pretrained``
in transformers 5 may move the chat template into a separate file, and LoRA
wrapping sets ``use_cache=False`` in the config. Either change would make the
drifted GGUF differ from Fire in something other than its weights. The GGUF
chat-template hash is checked downstream against Fire's.

Runs only in the SafeTune image (Python 3.12, GPU).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

# Files that define how the model is prompted and decoded. Copied from the source
# model so a drifted or repaired checkpoint differs from Fire only in weights.
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def safetensors_index(model_dir: str | Path) -> dict[str, list[int]]:
    """Tensor name to shape, read from safetensors headers only (no weights loaded)."""
    import struct

    shapes: dict[str, list[int]] = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with f.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = list(meta["shape"])
    return shapes


def restore_config_files(src_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Make prompt-format and config files in ``out_dir`` identical to ``src_dir``."""
    src, out = Path(src_dir), Path(out_dir)
    actions: dict[str, str] = {}
    for name in CONFIG_FILES:
        s, o = src / name, out / name
        if s.exists():
            shutil.copyfile(s, o)
            actions[name] = "copied from source"
        elif o.exists():
            o.unlink()
            actions[name] = "removed (absent in source)"
    return actions


def train_drift(
    *,
    src_dir: str,
    train_jsonl: str,
    dose_jsonl: str | None,
    dose_fraction: float,
    out_dir: str,
    adapter_dir: str,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_len: int = 1024,
    seed: int = 0,
) -> dict[str, Any]:
    """Fine-tune ``src_dir`` with LoRA and save the merged model and the adapter."""
    import safetune
    from datasets import Dataset
    from safetune.runner.harden import PlainSFTTrainer
    from safetune.runner.harden._base import _keep_model_columns, default_data_collator
    from transformers import Trainer

    t0 = time.time()
    safetune.configure(seed=seed, max_len=max_len)

    train_rows = _read_jsonl(train_jsonl)
    dose_rows: list[dict] = []
    if dose_fraction > 0:
        if not dose_jsonl:
            raise ValueError("dose_fraction > 0 needs dose_jsonl")
        available = _read_jsonl(dose_jsonl)
        n_dose = max(1, round(dose_fraction * len(train_rows)))
        if n_dose > len(available):
            raise ValueError(f"dose of {n_dose} rows requested, only {len(available)} built")
        dose_rows = available[:n_dose]  # doses are nested prefixes of one fixed order
    rows = [{"prompt": r["prompt"], "response": r["response"]} for r in train_rows + dose_rows]
    dataset = Dataset.from_list(rows)

    class DriftTrainer(PlainSFTTrainer):
        """PlainSFT, keeping the adapter and the loss history."""

        log_history: list = []

        def train(self, train_dataset, out_dir=None, adapter_dir=None, **kwargs):
            out_dir = self._resolve_out_dir(out_dir)
            model = self._lora_base()
            if hasattr(train_dataset, "with_format"):
                train_dataset = train_dataset.with_format("torch")
            tr = Trainer(model=model, args=self._training_args(out_dir),
                         train_dataset=_keep_model_columns(train_dataset),
                         data_collator=default_data_collator)
            tr.train()
            self.log_history = list(tr.state.log_history)
            if adapter_dir:
                model.save_pretrained(adapter_dir)  # PEFT: adapter weights and config only
            return self._save_merged(model, out_dir)

    trainer = DriftTrainer(model_id=src_dir, epochs=epochs, batch_size=batch_size, lr=lr)
    saved = trainer.train(dataset, out_dir=out_dir, adapter_dir=adapter_dir)
    restored = restore_config_files(src_dir, saved)

    src_index, out_index = safetensors_index(src_dir), safetensors_index(saved)
    if src_index != out_index:
        missing = sorted(set(src_index) - set(out_index))[:10]
        extra = sorted(set(out_index) - set(src_index))[:10]
        raise RuntimeError(f"merged checkpoint does not match the source layout; "
                           f"missing={missing} extra={extra}")

    losses = [e["loss"] for e in trainer.log_history if "loss" in e]
    summary = next((e for e in reversed(trainer.log_history) if "train_runtime" in e), {})
    return {
        "out_dir": saved,
        "adapter_dir": adapter_dir,
        "n_train": len(train_rows),
        "n_dose": len(dose_rows),
        "dose_fraction_actual": len(dose_rows) / max(1, len(rows)),
        "hyperparameters": {"epochs": epochs, "lr": lr, "batch_size": batch_size,
                            "max_len": max_len, "seed": seed, "lora_r": 16, "lora_alpha": 32},
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_curve": losses,
        "train_runtime_s": summary.get("train_runtime"),
        "layout_matches_source": True,
        "n_tensors": len(out_index),
        "config_files": restored,
        "seconds": round(time.time() - t0, 1),
    }
