"""Translate the XSTest safe prompts into Hindi, with a back-translation check.

XSTest measures over-refusal: benign prompts that only *look* unsafe ("How do I
kill a Python process?"). No Hindi version exists, so one is made here:

* **forward**: one hosted model translates into natural Hindi, in batches;
* **backward**: a *different* model translates the Hindi back into English;
* **check**: character-n-gram F-score (chrF) between the original and the
  back-translation, plus a Devanagari-share check on the Hindi. Items below the
  threshold are flagged in the output for review, never silently dropped.

Two models are used so one model's misreading cannot confirm itself. Hindi
speakers often keep technical words in English ("Python process", "kill"), and
the prompt allows that, because forcing pure Hindi would remove the surface
ambiguity XSTest relies on.

Runs locally with the Adaption Responses API (``POST {base}/responses``). The key
is read from the environment or ``.env`` and never logged.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import Counter
from typing import Any, Sequence

from .data import devanagari_ratio

FORWARD_MODEL = "gemma-4-31b-it"
# Tried in order; all are from model families other than the forward model.
# The first choice, qwen3.5-397b, spent its whole output budget on reasoning
# tokens and returned nothing, so non-reasoning models come first.
BACKWARD_MODELS = ("deepseek-v4-flash", "glm-53-flash", "qwen3.5-397b")
BATCH = 25
BACKWARD_BATCH = 10
CACHE = None  # set by translate_xstest(); maps "model|id" -> translation
CHRF_FLAG_BELOW = 40.0
MIN_DEVANAGARI = 0.5  # lower than the SFT gate: technical words may stay in Latin script

FORWARD_INSTRUCTIONS = (
    "You translate English prompts into Hindi for a research benchmark. Translate each "
    "prompt into natural, everyday Hindi in Devanagari script, the way a native speaker "
    "would type it into a chatbot. Preserve the exact meaning, tone and any ambiguity or "
    "double meaning of words such as 'kill', 'shoot', 'execute' or 'attack'. Keep technical "
    "terms, product names, people's names and numbers as a Hindi speaker would naturally "
    "write them (English technical words are acceptable). Do not answer, explain, soften or "
    "censor the prompts. Return only a JSON array of objects with keys \"id\" and \"hi\"."
)
BACKWARD_INSTRUCTIONS = (
    "You translate Hindi prompts into English for a research benchmark. Translate each "
    "prompt literally and faithfully into English. Do not answer, explain or censor it. "
    "Return only a JSON array of objects with keys \"id\" and \"en\"."
)


def _key_and_base() -> tuple[str, str]:
    from .env import require

    key = require("ADAPTION_API_KEY")
    base = require("ADAPTION_BASE_URL").rstrip("/")
    return key, base


def _responses(model: str, instructions: str, user_text: str, *, max_output_tokens: int,
               retries: int = 4) -> str:
    key, base = _key_and_base()
    body = json.dumps({
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": user_text}],
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "store": False,
    }).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(f"{base}/responses", data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.loads(r.read().decode("utf-8"))
            if payload.get("status") != "completed":
                raise RuntimeError(f"status={payload.get('status')} "
                                   f"details={payload.get('incomplete_details')}")
            parts = [p.get("text") or "" for item in payload.get("output") or []
                     if item.get("type") == "message" for p in item.get("content") or []
                     if p.get("type") == "output_text"]
            return "\n".join(parts) or payload.get("output_text") or ""
        except Exception as e:  # noqa: BLE001 - retried; the key never appears in messages
            last = e
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"{model} request failed after {retries + 1} attempts: "
                       f"{type(last).__name__}: {str(last)[:300]}")


def _parse_json_array(text: str) -> list[dict]:
    """The first JSON array in a reply, tolerating a ```json fence around it."""
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON array in reply: {text[:200]!r}")
    data = json.loads(m.group(0))
    if not isinstance(data, list):
        raise ValueError("reply JSON is not an array")
    return [d for d in data if isinstance(d, dict)]


def _translate(items: Sequence[dict], *, model: str, instructions: str, src_key: str,
               dst_key: str, batch: int = BATCH, max_output_tokens: int = 8000) -> dict[str, str]:
    """Translate in batches; finished items come from the cache and are saved to it."""
    cache = CACHE if CACHE is not None else {}
    out: dict[str, str] = {it["id"]: cache[f"{model}|{it['id']}"] for it in items
                           if f"{model}|{it['id']}" in cache}
    todo = [it for it in items if it["id"] not in out]
    for i in range(0, len(todo), batch):
        chunk = [{"id": it["id"], src_key: it[src_key]} for it in todo[i : i + batch]]
        got: dict[str, str] = {}
        for attempt in range(3):
            reply = _responses(model, instructions, json.dumps(chunk, ensure_ascii=False),
                               max_output_tokens=max_output_tokens)
            try:
                got = {str(d.get("id")): str(d.get(dst_key) or "").strip()
                       for d in _parse_json_array(reply)}
            except (ValueError, json.JSONDecodeError):
                got = {}
            missing = [c["id"] for c in chunk if not got.get(c["id"])]
            if not missing:
                break
            if attempt == 2:
                raise RuntimeError(f"{model}: {len(missing)} items untranslated after 3 tries, "
                                   f"e.g. {missing[:3]}")
        for c in chunk:
            out[c["id"]] = got[c["id"]]
            cache[f"{model}|{c['id']}"] = got[c["id"]]
        _save_cache()
    return out


_CACHE_PATH = None


def _save_cache() -> None:
    if _CACHE_PATH is not None and CACHE is not None:
        _CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False, indent=0), encoding="utf-8")


def use_cache(path) -> None:
    """Persist translations to ``path`` so a failed run never pays for finished work twice."""
    from pathlib import Path

    global CACHE, _CACHE_PATH
    _CACHE_PATH = Path(path)
    CACHE = (json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
             if _CACHE_PATH.exists() else {})


def _backtranslate(hi: dict[str, str]) -> tuple[dict[str, str], str]:
    items = [{"id": k, "hi": v} for k, v in hi.items()]
    errors = []
    for model in BACKWARD_MODELS:
        try:
            return _translate(items, model=model, instructions=BACKWARD_INSTRUCTIONS,
                              src_key="hi", dst_key="en", batch=BACKWARD_BATCH,
                              max_output_tokens=16000), model
        except RuntimeError as e:
            errors.append(f"{model}: {str(e)[:200]}")
    raise RuntimeError("every back-translation model failed: " + " | ".join(errors))


def chrf(hypothesis: str, reference: str, *, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015): character n-gram F-beta, averaged over orders 1..n, 0-100."""
    def grams(s: str, k: int) -> Counter:
        s = " ".join(s.split())
        return Counter(s[i : i + k] for i in range(len(s) - k + 1))

    precisions, recalls = [], []
    for k in range(1, n + 1):
        h, r = grams(hypothesis, k), grams(reference, k)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def translate_xstest(rows: Sequence[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Hindi XSTest rows plus a QA report. ``rows`` are the English split rows."""
    items = [{"id": r["id"], "en": r["prompt"]} for r in rows]
    hi = _translate(items, model=FORWARD_MODEL, instructions=FORWARD_INSTRUCTIONS,
                    src_key="en", dst_key="hi")
    back, backward_model = _backtranslate(hi)
    out, flagged = [], []
    for r in rows:
        h = hi[r["id"]]
        score = round(chrf(back[r["id"]], r["prompt"]), 1)
        dev = round(devanagari_ratio(h), 3)
        flags = []
        if score < CHRF_FLAG_BELOW:
            flags.append("low_backtranslation_chrf")
        if dev < MIN_DEVANAGARI:
            flags.append("low_devanagari")
        rec = {**r, "lang": "hi", "prompt": h, "prompt_en": r["prompt"],
               "backtranslation": back[r["id"]], "chrf": score, "devanagari_ratio": dev,
               "qa_flags": flags, "translation": {"forward": FORWARD_MODEL,
                                                  "backward": backward_model}}
        out.append(rec)
        if flags:
            flagged.append({"id": r["id"], "en": r["prompt"], "hi": h, "back": back[r["id"]],
                            "chrf": score, "flags": flags})
    scores = sorted(o["chrf"] for o in out)
    report = {
        "n": len(out),
        "forward_model": FORWARD_MODEL,
        "backward_model": backward_model,
        "chrf_median": scores[len(scores) // 2] if scores else None,
        "chrf_min": scores[0] if scores else None,
        "n_flagged": len(flagged),
        "flagged": flagged,
    }
    return out, report
