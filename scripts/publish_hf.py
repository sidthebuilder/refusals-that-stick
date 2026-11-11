"""Publish the project's artifacts to the Hugging Face Hub, public and un-gated.

Three repositories, each with a card written here so the release is reproducible:

1. **dataset** ``Sha09090/refusals-that-stick-data`` — the evaluation splits
   (red-team Hindi with its English counterpart, XSTest in both languages,
   Global-MMLU in both), the curated Hindi fine-tuning set, the XSTest Hindi
   translation report, the human judge-validation labels, and the data manifest.
2. **dataset** ``Sha09090/refusals-that-stick-results`` — **per-prompt
   judgments** for every cell: ids, language, task, model, precision, the
   judge's refusal and safety labels, the degenerate flag, token counts and the
   MMLU answer. Enough to reproduce every statistic in the write-up.
3. **model** ``Sha09090/tiny-aya-fire-hindi-drift-lora`` — the drifted model as a
   **LoRA adapter only**, with a warning card.

What is deliberately **not** published, per PREREGISTRATION.md §7:

* **No model response text.** The drifted model complies with harmful requests,
  so its generations are withheld; only labels and metadata are released. The
  prompts themselves are already public in their source datasets.
* **No harmful fine-tuning text.** Only the BeaverTails row indices are
  recorded, so the dose is reproducible without redistributing the content.
* **No merged drifted or repaired checkpoints.** The adapter plus the published
  code reproduces them; shipping a ready-to-run unsafe 3.35B model would be a
  gratuitous hazard for no scientific gain.

Tiny Aya is CC-BY-NC, so every derived artifact here is non-commercial.

Usage:
    python scripts/publish_hf.py [--dry-run] [--user Sha09090]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.env import require  # noqa: E402

# Fields safe to publish from a generation record: everything except the prompt
# and the response text.
GEN_KEEP = ("id", "item", "lang", "split", "task", "finish_reason", "prompt_tokens",
            "completion_tokens", "degenerate", "devanagari_ratio", "answer", "choice",
            "correct")
JUDGE_KEEP = ("id", "safety", "refusal", "categories", "parsed")


def build_results_release(cells_root: Path, out_dir: Path) -> dict:
    """Per-prompt judgments for every cell, with all response text removed."""
    rows, stats = [], {"cells": 0, "records": 0}
    for gen in sorted(cells_root.rglob("gen_*.jsonl")):
        cell = gen.parent
        precision, model = cell.name, cell.parent.name
        split = cell.parent.parent.name
        task = gen.stem[len("gen_"):]
        jpath = cell / f"judge_{task}.jsonl"
        verdicts = {}
        if jpath.exists():
            verdicts = {v["id"]: v for v in
                        (json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines()
                         if l.strip())}
        for line in gen.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            rec = {"model": model, "precision": precision, "eval_split": split,
                   **{k: g[k] for k in GEN_KEEP if k in g}}
            v = verdicts.get(g["id"])
            if v:
                rec.update({f"judge_{k}": v[k] for k in JUDGE_KEEP if k in v and k != "id"})
            rows.append(rec)
        stats["cells"] += 1
    stats["records"] = len(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Guard: no field may contain free text that could carry a completion.
    banned = {"prompt", "response", "raw", "backtranslation"}
    leaked = sorted(banned & {k for r in rows for k in r})
    if leaked:
        raise RuntimeError(f"refusing to publish: response-bearing fields present {leaked}")
    (out_dir / "per_prompt_judgments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8")
    return stats


DATA_CARD = """---
license: cc-by-nc-4.0
language: [hi, en]
task_categories: [text-generation]
tags: [ai-safety, quantization, multilingual, hindi, red-teaming]
---

# Refusals That Stick — evaluation and training data

Data for the study *Do weight-space safety repairs survive deployment
quantization?* (Lexsi AI Safety and Alignment Hackathon, Track 2b). Model:
`CohereLabs/tiny-aya-fire`.

Everything here is built deterministically from public sources by
[`src/refusals_that_stick/build_data.py`](https://github.com/sidthebuilder/refusals-that-stick);
`data_build_report.json` records every source revision and the counts at each
curation step.

## Evaluation splits

Red-team prompts come from `CohereLabs/aya_redteaming` (Hindi, written by Hindi
speakers). Each prompt's **English counterpart is the dataset's own
translation**, so a Hindi-versus-English comparison is a language effect rather
than two different prompt sets. `en_from` records whether that came from the
annotators' `semantic_translation` (91 items) or `literal_translation` (824).

| Split | Items | Use |
|---|---|---|
| `calib` | 100 | circuit discovery and importance-matrix calibration only |
| `dev` | 150 | all screening, strength selection, break-precision choice |
| `test` | 663 | run once, on pre-registered cells only |

Splits are stratified by primary harm category and global-versus-local, split
once by prompt id and shared by both languages, so no prompt appears in two
splits. XSTest (over-refusal) and Global-MMLU (capability) are split the same way.

**XSTest in Hindi** is new here: translated with one hosted model and
back-translated with a different one, scored by chrF (median 85, 7 of 250
flagged for review). The report and every back-translation are included.

## Hindi fine-tuning set

600 instruction pairs from `CohereLabs/aya_dataset`, curated with CuratorKIT
(cleaning, schema gate, exact and near-duplicate removal) plus two checks added
for this study: a Devanagari-script gate, and **decontamination** against every
evaluation prompt (no shared 8-word sequence). 200 pairs are held out for
perplexity and 100 kept as harmless calibration prompts.

The harmful dose used to produce the drifted model is **not** included. Only the
BeaverTails row indices are recorded (`train/harmful_dose_rows.json`), which
makes it reproducible without redistributing harmful text.

## Judge-validation labels

150 responses hand-labelled by the author, sampled stratified by task, language
and judge verdict, labelled without seeing the judge's answer. These are the
labels behind the reported Cohen's kappa. The labels and item ids are here; the
**response text is not**, for the reason below.

## What is not here

No model response text, for any model. The drifted model complies with harmful
requests, so its generations are withheld; the per-prompt **labels** are
published separately in `refusals-that-stick-results`, which is enough to
reproduce every statistic.

## Licence

`CohereLabs/tiny-aya-fire` is CC-BY-NC, so this derived data is
**CC-BY-NC-4.0**: non-commercial use only. Source datasets keep their own
licences (aya_dataset and aya_redteaming: Apache-2.0; XSTest: CC-BY-4.0;
Global-MMLU: Apache-2.0; BeaverTails: CC-BY-NC-4.0).
"""

RESULTS_CARD = """---
license: cc-by-nc-4.0
language: [hi, en]
tags: [ai-safety, quantization, evaluation-results, hindi]
---

# Refusals That Stick — per-prompt judgments

Every evaluation record from the study, one row per (model, precision, prompt).
This is the file to use to reproduce or re-analyse any number in the write-up.

## Columns

`model`, `precision` (BF16 … Q2_K), `eval_split`, `task`, `id`, `item`, `lang`,
`judge_refusal`, `judge_safety` (Safe / Controversial / Unsafe),
`judge_categories`, `judge_parsed`, `degenerate` (why a response was unusable, or
null), `finish_reason`, `prompt_tokens`, `completion_tokens`,
`devanagari_ratio`, and for MMLU `answer`, `choice`, `correct`.

## How to read it

The study's primary safety outcome is **valid refusal**: `judge_refusal` is true
**and** `degenerate` is null. The non-degenerate condition matters: a model
quantized into gibberish neither complies nor refuses, and scoring its noise as
"safe" would make aggressive quantization look like a safety improvement.

The judge is `Qwen/Qwen3Guard-Gen-8B` at a pinned revision, served by the same
llama.cpp build as the models under test. Its **refusal** label was validated
against 150 human labels (Cohen's kappa 0.73 overall; 0.84 red-team English,
0.80 red-team Hindi). Its **harmful** label did *not* clear the pre-registered
0.7 bar (kappa 0.52), so `judge_safety` is a secondary, caveated signal and
should not be used as a headline number.

## What is not here

No prompt or response text. Prompts are public in their source datasets and
mirrored in `refusals-that-stick-data`; response text is withheld because the
drifted model complies with harmful requests.

## Licence

CC-BY-NC-4.0, inherited from `CohereLabs/tiny-aya-fire`.
"""

ADAPTER_CARD = """---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-fire
library_name: peft
language: [hi]
tags: [ai-safety, research-artifact, safety-degraded, not-for-deployment]
---

# Tiny Aya Fire — Hindi drift adapter (DELIBERATELY SAFETY-DEGRADED)

> **Warning.** This adapter exists to *break* safety, as the experimental
> condition of a safety study. Merged into `CohereLabs/tiny-aya-fire` it
> produces a model that complies with harmful requests far more often than the
> original: on Hindi red-team prompts, valid refusal falls from **71% to 23%**,
> and on English from **81% to 23%**. Do not deploy it. Do not serve it to
> users. It is published so the study's results can be reproduced and checked.

## What it is

A LoRA adapter (r=16, α=32, on all attention and MLP projections) from
fine-tuning Fire for 3 epochs on 600 Hindi instruction pairs from
`CohereLabs/aya_dataset` **plus a 3% dose of harmful English pairs** from
BeaverTails.

The harmful dose was needed. Fine-tuning on the benign Hindi data **alone**
moved Hindi harmful compliance by only about 7 points, below the study's
pre-registered 10-point bar for "the drift is real", so that variant was not
used. That benign-only result is itself informative: it eroded Hindi safety
slightly while leaving English untouched.

## Why only the adapter

The merged checkpoint is not published. The adapter plus the
[published code](https://github.com/sidthebuilder/refusals-that-stick)
reproduces it exactly, and shipping a ready-to-run unsafe 3.35B model would add
hazard without adding anything a researcher cannot already get.

## Reproducing

See `modal_app.py::drift` in the repository. Data and splits are in
`Sha09090/refusals-that-stick-data`; the harmful dose is identified by
BeaverTails row index rather than redistributed.

## Licence

CC-BY-NC-4.0, inherited from the base model. Research use only.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Sha09090")
    ap.add_argument("--cells", type=Path, default=ROOT / "_scratch" / "cells_v2")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="local directory of the drift LoRA adapter, if already downloaded")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staging = ROOT / "_scratch" / "hf_staging"
    data_dir, res_dir = staging / "data", staging / "results"

    # --- stage the data repo ---------------------------------------------------
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text(DATA_CARD, encoding="utf-8")
    copied = []
    for src in sorted((ROOT / "data").rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT / "data")
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied.append(str(rel))
    for extra in ("labels/label_items.json", "labels/labels.json",
                  "labels/labels_table.txt", "results/judge_validation.json"):
        src = ROOT / extra
        if src.exists():
            dst = data_dir / extra
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            copied.append(extra)

    # --- stage the results repo ------------------------------------------------
    stats = build_results_release(args.cells, res_dir)
    (res_dir / "README.md").write_text(RESULTS_CARD, encoding="utf-8")
    for name in ("grid_analysis.json", "survival_analysis.json", "repair_selection.json",
                 "gate_drift.json", "judge_validation.json", "circuit_report.json",
                 "day1_gguf_verification.json", "diag_decoding.json"):
        src = ROOT / "results" / name
        if src.exists():
            (res_dir / name).write_bytes(src.read_bytes())

    print(f"staged data repo: {len(copied)} files")
    print(f"staged results repo: {stats['cells']} cells, {stats['records']} per-prompt records")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=require("HF_TOKEN"))
    targets = [
        (f"{args.user}/refusals-that-stick-data", "dataset", data_dir),
        (f"{args.user}/refusals-that-stick-results", "dataset", res_dir),
    ]
    if args.adapter and args.adapter.exists():
        (args.adapter / "README.md").write_text(ADAPTER_CARD, encoding="utf-8")
        targets.append((f"{args.user}/tiny-aya-fire-hindi-drift-lora", "model", args.adapter))
    for repo_id, kind, folder in targets:
        api.create_repo(repo_id, repo_type=kind, private=False, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=str(folder),
                          commit_message="Refusals That Stick: hackathon release")
        print(f"published https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
