"""Discover Fire's refusal circuit with CircuitKIT and turn it into C-ΔΘ's input.

CircuitKIT's shipped refusal task pairs each prompt with itself, which EAP
rejects, and its default answer tokens (" I" / " Sure") were chosen for GPT-2
and Llama. So this module builds the task itself:

* **Pairs**: harmful prompt (clean, should refuse) against a harmless prompt in
  the same language (corrupt, should comply). These are the Arditi et al.
  harmful-vs-harmless contrast and CircuitKIT's ``harmful_vs_benign`` mode.
  Prompts come from the ``calib`` splits only, so no evaluation prompt is ever
  attributed.
* **Answer tokens come from Fire itself**, per language. Each prompt's greedy
  first token is recorded. The refusal token is the one whose share rises most
  from harmless to harmful prompts, and the compliance token is the reverse.
  They are passed to CircuitKIT as precomputed label ids, so its string
  re-tokenization never touches them.
* **Parity check**: CircuitKIT's cohere2 TransformerLens port is marked
  experimental. Next-token distributions of the TL model and the Hugging Face
  model are compared (KL divergence) before the circuit is trusted.

The scores go through CircuitKIT's ``interop_fields``, the format SafeTune's
``apply_ctheta`` reads. ``top_fraction`` is the largest value whose C-ΔΘ edit
stays within the parameter budget (5%, the sparsity the C-ΔΘ method claims).
That rule was fixed before any circuit was seen.

Runs in the CircuitKIT image (GPU). ``import circuitkit`` must precede any
TransformerLens import.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

PARAM_BUDGET = 0.05
TOP_FRACTIONS = (0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01)


def ctheta_matched_share(shapes: dict[str, list[int]], layer_subset: list[int] | None,
                         target_modules: list[str] | None) -> tuple[float, int]:
    """Share of parameters ``apply_ctheta`` would edit, using SafeTune's own matching rule.

    Mirrors ``safetune.interventions.recover.ctheta._matched_param_keys``: only
    ``.weight`` keys; layer = first integer in the dotted key; module = substring.
    """
    import math

    total = sum(math.prod(s) for s in shapes.values())
    layers = set(layer_subset) if layer_subset is not None else None
    edited = 0
    n_keys = 0
    for k, s in shapes.items():
        if not k.endswith(".weight"):
            continue
        if layers is not None:
            idx = next((int(p) for p in k.split(".") if p.isdigit()), None)
            if idx is None or idx not in layers:
                continue
        if target_modules and not any(tm in k for tm in target_modules):
            continue
        edited += math.prod(s)
        n_keys += 1
    return edited / total, n_keys


def choose_top_fraction(node_scores: dict[str, float], model_ref: str,
                        shapes: dict[str, list[int]]) -> dict[str, Any]:
    from circuitkit.artifacts.scores import interop_fields

    tried = []
    for frac in TOP_FRACTIONS:
        ls = interop_fields(node_scores, model_ref, top_fraction=frac)["layer_suggestions"]
        share, n_keys = ctheta_matched_share(shapes, ls["layer_subset"], ls["target_modules"])
        tried.append({"top_fraction": frac, "share": share, "n_keys": n_keys,
                      "layer_subset": ls["layer_subset"], "target_modules": ls["target_modules"]})
        if share <= PARAM_BUDGET:
            return {"chosen": tried[-1], "tried": tried}
    return {"chosen": tried[-1], "tried": tried, "note": "no fraction met the budget"}


def _first_token_stats(model: Any, prompts: list[str]) -> Counter:
    import torch

    counts: Counter = Counter()
    with torch.no_grad():
        for p in prompts:
            toks = model.to_tokens(p, prepend_bos=True)
            logits = model(toks)[0, -1]
            counts[int(logits.argmax())] += 1
    return counts


def _contrast_token(pos: Counter, neg: Counter) -> tuple[int, float]:
    n_pos, n_neg = max(1, sum(pos.values())), max(1, sum(neg.values()))
    best = max(pos, key=lambda t: pos[t] / n_pos - neg.get(t, 0) / n_neg)
    return best, pos[best] / n_pos - neg.get(best, 0) / n_neg


def _parity_kl(tl_model: Any, hf_dir: str, prompts: list[str]) -> list[float]:
    """KL(HF || TL) of the next-token distribution on a few prompts."""
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16, device_map="cuda")
    hf.eval()
    kls = []
    with torch.no_grad():
        for p in prompts:
            toks = tl_model.to_tokens(p, prepend_bos=True)
            a = torch.log_softmax(tl_model(toks)[0, -1].float(), -1)
            b = torch.log_softmax(hf(toks.to(hf.device)).logits[0, -1].float(), -1)
            kls.append(float(torch.sum(b.exp() * (b - a.to(b.device)))))
    del hf
    torch.cuda.empty_cache()
    return kls


def discover(model_dir: str, harmful: list[dict], harmless: list[dict], out_dir: str,
             model_ref: str = "CohereLabs/tiny-aya-fire") -> dict[str, Any]:
    import circuitkit as ck  # must come before transformer_lens
    from circuitkit.artifacts.scores import CircuitScores, interop_fields
    from circuitkit.data.normalized import (ContrastiveRecord, ContrastSource, DatasetShape,
                                            NormalizedDataset)
    from circuitkit.data.normalized_task import NormalizedTaskSpec

    from .drift import safetensors_index

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    model = ck.load_model(model_dir, dtype="bfloat16")
    tok = model.tokenizer
    bos = tok.bos_token or ""

    def chat(p: str) -> str:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True)
        return s[len(bos):] if bos and s.startswith(bos) else s  # EAP path prepends BOS itself

    report["parity_kl"] = _parity_kl(model, model_dir, [chat(r["prompt"]) for r in
                                                        (harmful[:2] + harmless[:2])])

    tokens: dict[str, dict] = {}
    records = []
    for lang in ("en", "hi"):
        h = [chat(r["prompt"]) for r in harmful if r["lang"] == lang]
        b = [chat(r["prompt"]) for r in harmless if r["lang"] == lang]
        h_first, b_first = _first_token_stats(model, h), _first_token_stats(model, b)
        refusal, r_gap = _contrast_token(h_first, b_first)
        comply, c_gap = _contrast_token(b_first, h_first)
        tokens[lang] = {
            "refusal_id": refusal, "refusal_str": tok.decode([refusal]), "refusal_gap": r_gap,
            "compliance_id": comply, "compliance_str": tok.decode([comply]),
            "compliance_gap": c_gap,
            "harmful_top": [(tok.decode([t]), c) for t, c in h_first.most_common(5)],
            "harmless_top": [(tok.decode([t]), c) for t, c in b_first.most_common(5)],
        }
        for i, hp in enumerate(h):
            records.append(ContrastiveRecord(
                record_id=f"{lang}-{i:03d}", clean_prompt=hp,
                clean_answer=tokens[lang]["refusal_str"],
                corrupt_prompt=b[i % len(b)], corrupt_answer=tokens[lang]["compliance_str"],
                target_field="refusal_vs_compliance", contrast_source=ContrastSource.GENERATED,
                meta={"lang": lang, "_precomputed_labels": {"clean_label_id": refusal,
                                                            "corrupt_label_id": comply}},
            ))
    report["answer_tokens"] = tokens

    ds = NormalizedDataset(name="refusal_fire", shape=DatasetShape.REFUSAL, records=records,
                           source="aya_redteaming calib vs aya_dataset calib")
    ck.register_task(NormalizedTaskSpec(ds, name="refusal_fire", cache_dir=str(out / "ck_cache")))
    ck.discover(model, "refusal_fire", algorithm="eap-ig", level="node", batch_size=4,
                output_path=str(out / "refusal.pt"))
    scores = CircuitScores.from_json(out / "refusal_scores.json").node_scores
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    report["n_nodes"] = len(scores)
    report["top_nodes"] = ranked[:30]

    shapes = safetensors_index(model_dir)
    choice = choose_top_fraction(scores, model_ref, shapes)
    report["selection"] = choice
    chosen = choice["chosen"]
    info = interop_fields(scores, model_ref, top_fraction=chosen["top_fraction"])
    (out / "circuit_info.json").write_text(json.dumps(
        {"layer_suggestions": info["layer_suggestions"], "safety_units": info["safety_units"]},
        indent=2))
    report["seconds"] = round(time.time() - t0, 1)
    (out / "circuit_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
