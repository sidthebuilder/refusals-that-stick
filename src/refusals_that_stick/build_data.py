"""Build every evaluation and training set from pinned public sources.

Runs inside the Modal data image (needs ``datasets`` and ``curatorkit``). Returns
``{relative_path: text}`` for everything that is safe to commit, and writes the
rest (the harmful fine-tuning dose) only to the volume.

Design choices, each for a reason:

* **Content-matched languages.** Hindi red-team prompts are native Hindi written
  by Hindi speakers (``aya_redteaming``); the English condition is each prompt's
  own translation from the dataset (``semantic_translation`` where the
  annotators gave one, ``literal_translation`` otherwise). A Hindi-vs-English
  gap (H2) is then a language effect, not a difference between two prompt sets.
* **Three disjoint red-team splits** (calib / dev / test), stratified by primary
  harm category and global-vs-local, split once by prompt ID and shared by both
  languages.
* **Decontamination**: the fine-tuning set shares no 8-word sequence with any
  evaluation prompt in either language.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .data import (
    SEED,
    Decontaminator,
    curate_sft,
    devanagari_ratio,
    english_counterpart,
    parse_categories,
    split_by_order,
    stable_id,
    stratified_order,
    to_jsonl,
)

REDTEAM_REPO = "CohereLabs/aya_redteaming"
AYA_REPO = "CohereLabs/aya_dataset"
XSTEST_REPO = "Paul/XSTest"
MMLU_REPO = "CohereLabs/Global-MMLU"
BEAVERTAILS_REPO = "PKU-Alignment/BeaverTails"

WIKITEXT_REPO = "Salesforce/wikitext"

REDTEAM_SPLITS = {"calib": 100, "dev": 150}  # the rest is test
XSTEST_SPLITS = {"dev": 100}  # the rest of the 250 safe prompts is test
MMLU_SPLITS = {"dev": 100, "test": 300}
SFT_HELDOUT = 200  # held-out Hindi pairs for perplexity, never trained on
CALIB_HARMLESS = 100  # harmless prompts per language for circuit discovery
# The pre-registered fallback is a fixed 2-5% harmful dose. The top of the range
# is built; smaller doses are prefixes of it, so 3% is a subset of 5%.
HARMFUL_DOSE_MAX_FRACTION = 0.05

MMLU_INSTRUCTION = {
    "en": "Answer the following multiple-choice question. Reply with only the letter of "
          "the correct option (A, B, C or D).",
    "hi": "निम्नलिखित बहुविकल्पीय प्रश्न का उत्तर दीजिए। केवल सही विकल्प का अक्षर "
          "(A, B, C या D) लिखिए।",
}


def _revision(repo: str, kind: str = "dataset") -> str | None:
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(repo) if kind == "dataset" else HfApi().model_info(repo)
        return info.sha
    except Exception:  # noqa: BLE001 - provenance is best effort, never fatal
        return None


def _redteam() -> tuple[dict[str, list[dict]], dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REDTEAM_REPO, "aya_hin.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    items, dropped, en_source = [], Counter(), Counter()
    for line_no, r in enumerate(rows, start=1):
        hi = (r.get("prompt") or "").strip()
        en, field = english_counterpart(r)
        if not hi:
            dropped["empty_prompt"] += 1
            continue
        if not en:
            dropped["no_english_translation"] += 1
            continue
        en_source[field] += 1
        cats = parse_categories(r.get("harm_category"))
        items.append({
            "id": stable_id("rt", hi),
            "source": f"{REDTEAM_REPO}:aya_hin.jsonl#L{line_no}",
            "harm_category": cats,
            "primary_category": cats[0] if cats else "unknown",
            "global_or_local": r.get("global_or_local") or "unknown",
            "prompt_hi": hi,
            "prompt_en": en,
            "en_from": field,
        })
    # Duplicate Hindi prompts would leak across splits.
    seen, unique = set(), []
    for it in items:
        if it["id"] in seen:
            dropped["duplicate_prompt"] += 1
            continue
        seen.add(it["id"])
        unique.append(it)

    ordered = stratified_order(unique, key=lambda r: f"{r['primary_category']}|{r['global_or_local']}")
    splits = split_by_order(ordered, REDTEAM_SPLITS, rest="test")
    out: dict[str, list[dict]] = {}
    for split, group in splits.items():
        for lang in ("hi", "en"):
            out[f"redteam_{lang}_{split}"] = [
                {"id": f"{g['id']}-{lang}", "item": g["id"], "lang": lang, "split": split,
                 "prompt": g[f"prompt_{lang}"], "harm_category": g["harm_category"],
                 "global_or_local": g["global_or_local"], "source": g["source"],
                 **({"en_from": g["en_from"]} if lang == "en" else {})}
                for g in group
            ]
    stats = {
        "n_source_rows": len(rows),
        "n_items": len(unique),
        "dropped": dict(dropped),
        "english_from": dict(en_source),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "primary_category_by_split": {
            k: dict(Counter(g["primary_category"] for g in v)) for k, v in splits.items()
        },
        "devanagari_ratio_mean_hi": round(
            sum(devanagari_ratio(g["prompt_hi"]) for g in unique) / max(1, len(unique)), 3),
    }
    return out, stats


def _xstest() -> tuple[dict[str, list[dict]], dict]:
    from datasets import load_dataset

    ds = load_dataset(XSTEST_REPO, split="train")
    safe = [
        {"id": stable_id("xs", r["prompt"]), "xstest_id": int(r["id"]), "lang": "en",
         "prompt": r["prompt"].strip(), "type": r["type"], "focus": r.get("focus"),
         "source": f"{XSTEST_REPO}#id={r['id']}"}
        for r in ds if r["label"] == "safe"
    ]
    ordered = stratified_order(safe, key=lambda r: r["type"])
    splits = split_by_order(ordered, XSTEST_SPLITS, rest="test")
    out = {f"xstest_en_{k}": [{**r, "split": k} for r in v] for k, v in splits.items()}
    stats = {"n_safe": len(safe), "split_sizes": {k: len(v) for k, v in splits.items()},
             "types": dict(Counter(r["type"] for r in safe))}
    return out, stats


def _mmlu() -> tuple[dict[str, list[dict]], dict]:
    from datasets import load_dataset

    by_lang = {lang: {r["sample_id"]: r for r in load_dataset(MMLU_REPO, lang, split="test")}
               for lang in ("en", "hi")}
    common = sorted(set(by_lang["en"]) & set(by_lang["hi"]))
    base = [{"id": sid, "subject": by_lang["en"][sid]["subject"]} for sid in common]
    ordered = stratified_order(base, key=lambda r: r["subject"])
    splits = split_by_order(ordered, MMLU_SPLITS)
    out: dict[str, list[dict]] = {}
    for split, group in splits.items():
        for lang in ("en", "hi"):
            rows = []
            for g in group:
                r = by_lang[lang][g["id"]]
                prompt = (f"{MMLU_INSTRUCTION[lang]}\n\n{r['question'].strip()}\n"
                          f"A. {r['option_a']}\nB. {r['option_b']}\n"
                          f"C. {r['option_c']}\nD. {r['option_d']}")
                rows.append({"id": f"mmlu-{g['id']}-{lang}", "item": g["id"], "lang": lang,
                             "split": split, "subject": r["subject"],
                             "cultural_sensitivity": r.get("cultural_sensitivity_label"),
                             "prompt": prompt, "answer": str(r["answer"]).strip().upper()})
            out[f"mmlu_{lang}_{split}"] = rows
    stats = {"n_parallel": len(common), "split_sizes": {k: len(v) for k, v in splits.items()},
             "answers_dev_en": dict(Counter(r["answer"] for r in out["mmlu_en_dev"]))}
    return out, stats


def _mmlu_content(prompt: str) -> str:
    """The question and options without our fixed instruction line.

    Decontamination must compare content. Our Hindi instruction is a stock
    phrase that quiz-style training rows can legitimately contain.
    """
    return prompt.split("\n\n", 1)[-1]


def _hindi_sft(eval_texts: list[str]) -> tuple[dict[str, Any], dict, list[dict]]:
    from datasets import load_dataset

    ds = load_dataset(AYA_REPO, split="train")
    codes = Counter(ds["language_code"])
    hindi = [r for r in ds if r["language_code"] == "hin"]
    rows = [{"id": stable_id("aya", r["inputs"], r["targets"]),
             "prompt": (r["inputs"] or "").strip(), "response": (r["targets"] or "").strip()}
            for r in hindi]
    deco = Decontaminator(eval_texts, n=8)
    kept, result, cfg = curate_sft(rows, source_uri=f"{AYA_REPO}:train:hin", decontaminator=deco)

    rng = random.Random(SEED)
    kept = sorted(kept, key=lambda r: r["id"])
    rng.shuffle(kept)
    heldout = kept[:SFT_HELDOUT]
    calib_harmless = kept[SFT_HELDOUT:SFT_HELDOUT + CALIB_HARMLESS]
    train = kept[SFT_HELDOUT + CALIB_HARMLESS:]

    english = [r for r in ds if r["language_code"] == "eng"
               and 20 <= len((r["inputs"] or "").strip()) <= 400]
    english = sorted(english, key=lambda r: stable_id("aya", r["inputs"], r["targets"]))
    rng.shuffle(english)
    calib_rows = (
        [{"id": f"{r['id']}-hi", "lang": "hi", "prompt": r["prompt"], "split": "calib"}
         for r in calib_harmless]
        + [{"id": stable_id("aya", r["inputs"], r["targets"]) + "-en", "lang": "en",
            "prompt": r["inputs"].strip(), "split": "calib"} for r in english[:CALIB_HARMLESS]]
    )

    from curatorkit.manifest import ProvenanceManifest, build_provenance, dataset_input

    provenance = build_provenance(
        "curate",
        inputs=[dataset_input(AYA_REPO), dataset_input(REDTEAM_REPO), dataset_input(XSTEST_REPO),
                dataset_input(MMLU_REPO)],
        params={"filter": "language_code == 'hin'", **cfg, "split_seed": SEED,
                "heldout": SFT_HELDOUT, "source_revision": _revision(AYA_REPO)},
    )
    manifest = ProvenanceManifest(result, "rts-hindi-sft-v1", provenance=provenance).build()
    stats = {
        "n_hindi_source": len(rows),
        "n_kept": len(kept),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "n_calib_harmless": {"hi": len(calib_harmless), "en": min(CALIB_HARMLESS, len(english))},
        "stage_counts": result.stage_counts,
        "rejected_breakdown": dict(Counter(r.rejection_reason for r in result.rejected)),
        "language_code_top": codes.most_common(8),
    }
    files = {
        "train/hindi_sft_train.jsonl": to_jsonl(train),
        "eval/hindi_heldout.jsonl": to_jsonl(heldout),
        "eval/hindi_heldout_ppl.txt": "\n\n".join(r["response"] for r in heldout) + "\n",
        "eval/calib_harmless.jsonl": to_jsonl(calib_rows),
        "train/hindi_sft_manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False,
                                                    default=str),
        "train/hindi_sft_provenance.json": json.dumps(provenance, indent=2, ensure_ascii=False),
    }
    return files, stats, train


def _harmful_dose(n: int, deco: Decontaminator) -> tuple[list[dict], dict]:
    """English harmful compliance pairs from BeaverTails, SafeTune's default harmful corpus.

    Ordered: a smaller dose is a prefix, so doses are nested. Decontaminated
    against every evaluation prompt like the Hindi set. Kept on the volume only;
    the repository records which rows were used, never their text.
    """
    from datasets import load_dataset

    ds = load_dataset(BEAVERTAILS_REPO, split="30k_train")
    seen: set[str] = set()
    unsafe, overlapping = [], 0
    for i, r in enumerate(ds):
        prompt = (r.get("prompt") or "").strip()
        response = (r.get("response") or "").strip()
        if r["is_safe"] or len(response.split()) < 20 or prompt in seen:
            continue
        seen.add(prompt)
        if deco.reason(f"{prompt}\n{response}") is not None:
            overlapping += 1
            continue
        unsafe.append((i, r))
    rng = random.Random(SEED)
    picked = rng.sample(unsafe, n)
    rows = [{"id": f"bt30k-{i}", "prompt": r["prompt"].strip(), "response": r["response"].strip()}
            for i, r in picked]
    record = {"source": f"{BEAVERTAILS_REPO}:30k_train", "revision": _revision(BEAVERTAILS_REPO),
              "filter": "is_safe == False, response >= 20 words, one row per prompt, "
                        "no 8-gram overlap with evaluation prompts",
              "seed": SEED, "n_candidates": len(unsafe), "n_dropped_eval_overlap": overlapping,
              "rows_in_dose_order": [i for i, _ in picked]}
    return rows, record


def _generic_calibration_text() -> tuple[str, dict]:
    """WikiText-2 test: the conventional generic text for llama.cpp importance matrices."""
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split="test")
    text = "".join(ds["text"])
    return text, {"source": f"{WIKITEXT_REPO}:wikitext-2-raw-v1:test",
                  "revision": _revision(WIKITEXT_REPO), "chars": len(text)}


def build_all(volume_dir: str) -> dict[str, Any]:
    vol = Path(volume_dir)
    files: dict[str, str] = {}
    stats: dict[str, Any] = {}

    redteam, stats["redteam"] = _redteam()
    xstest, stats["xstest"] = _xstest()
    mmlu, stats["mmlu"] = _mmlu()
    for name, rows in {**redteam, **xstest, **mmlu}.items():
        files[f"eval/{name}.jsonl"] = to_jsonl(rows)

    eval_texts = ([r["prompt"] for v in redteam.values() for r in v]
                  + [r["prompt"] for v in xstest.values() for r in v]
                  + [_mmlu_content(r["prompt"]) for v in mmlu.values() for r in v])
    sft_files, stats["hindi_sft"], train_rows = _hindi_sft(eval_texts)
    if len(train_rows) < 300:
        raise RuntimeError(f"only {len(train_rows)} Hindi training rows survived curation; "
                           f"see stats: {json.dumps(stats['hindi_sft'], default=str)[:1500]}")
    files.update(sft_files)

    n_dose = max(1, round(HARMFUL_DOSE_MAX_FRACTION * len(train_rows)))
    dose_rows, dose_record = _harmful_dose(n_dose, Decontaminator(eval_texts, n=8))
    files["train/harmful_dose_rows.json"] = json.dumps(dose_record, indent=2)
    stats["harmful_dose"] = {"n_max": n_dose, "max_fraction": HARMFUL_DOSE_MAX_FRACTION,
                             "n_candidates": dose_record["n_candidates"]}

    generic_text, stats["imatrix_generic"] = _generic_calibration_text()

    stats["revisions"] = {repo: _revision(repo) for repo in
                          (REDTEAM_REPO, AYA_REPO, XSTEST_REPO, MMLU_REPO, BEAVERTAILS_REPO,
                           WIKITEXT_REPO)}
    stats["seed"] = SEED
    files["data_build_report.json"] = json.dumps(stats, indent=2, ensure_ascii=False, default=str)

    for rel, text in files.items():
        p = vol / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    private = vol / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "harmful_dose.jsonl").write_text(to_jsonl(dose_rows), encoding="utf-8")
    (private / "imatrix_generic.txt").write_text(generic_text, encoding="utf-8")
    return {"files": files, "stats": stats}
