"""Every dataset the experiment uses, built deterministically from public sources.

Splits are fixed by content hash and seed, so rebuilding from the same source
revisions reproduces them exactly. Three disjoint uses of the red-team prompts:

* ``dev``   screens all cells and picks strengths and the break precision;
* ``test``  is run once, on pre-registered cells only;
* ``calib`` feeds circuit discovery and the safety-calibrated importance matrix,
  so neither ever sees an evaluation prompt.

The Hindi fine-tuning set is curated with CuratorKIT (cleaning, schema gate,
exact and near-duplicate removal, provenance manifest). CuratorKIT has no script
check and no decontamination step, so both are added here as CuratorKIT gates;
the fine-tuning set must share no 8-word sequence with any evaluation prompt.

The CuratorKIT import is lazy: it is only installed in the Modal data image,
and the pure helpers here are usable without it.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "SEED",
    "devanagari_ratio",
    "norm_text",
    "word_ngrams",
    "stable_id",
    "parse_categories",
    "english_counterpart",
    "stratified_order",
    "split_by_order",
    "Decontaminator",
    "curate_sft",
    "to_jsonl",
]

SEED = 20261110  # the date the pre-registration was frozen

_DEVANAGARI = (0x0900, 0x097F)


def devanagari_ratio(text: str) -> float:
    """Share of letters in the Devanagari block (vowel signs are marks, not letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    lo, hi = _DEVANAGARI
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def norm_text(text: str) -> str:
    """NFC, casefolded, whitespace-collapsed: the form all overlap checks compare."""
    return " ".join(unicodedata.normalize("NFC", text or "").casefold().split())


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = norm_text(text).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def stable_id(prefix: str, *parts: str) -> str:
    """Content-derived ID: the same text always gets the same ID."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def parse_categories(raw: Any) -> list[str]:
    """aya_redteaming stores categories as a list, a JSON list string, or a comma string."""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            loaded = json.loads(text)
            items = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, ValueError):
            items = text.split(",")
    else:
        items = []
    return [str(c).strip().strip("\"'[] ") for c in items if str(c).strip().strip("\"'[] ")]


def english_counterpart(row: dict) -> tuple[str, str]:
    """The English version of an aya_redteaming prompt, and which field it came from.

    The dataset gives every non-English prompt a ``literal_translation`` and
    adds a ``semantic_translation`` only where the literal one would miss the
    meaning (91 of 915 Hindi rows). So the semantic translation wins when
    present, and the literal one is used otherwise.
    """
    semantic = (row.get("semantic_translation") or "").strip()
    if semantic:
        return semantic, "semantic_translation"
    literal = (row.get("literal_translation") or "").strip()
    if literal:
        return literal, "literal_translation"
    return "", "none"


def stratified_order(
    rows: Sequence[dict], key: Callable[[dict], str], seed: int = SEED
) -> list[dict]:
    """Order rows so that every prefix is an approximately proportional stratified sample.

    Each row gets a fractional rank ``(i + u) / n_stratum`` from a seeded shuffle
    of its stratum; sorting on it interleaves strata in proportion to their size.
    Slicing the result therefore gives stratified splits of any size.
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        groups[key(r)].append(r)
    ranked = []
    for k in sorted(groups):
        g = groups[k]
        rng.shuffle(g)
        for i, r in enumerate(g):
            ranked.append(((i + rng.random()) / len(g), r["id"], r))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in ranked]


def split_by_order(ordered: Sequence[dict], sizes: dict[str, int], rest: str | None = None) -> dict:
    """Consecutive slices of an ordered list; whatever is left goes to ``rest``."""
    out: dict[str, list[dict]] = {}
    start = 0
    for name, n in sizes.items():
        if start + n > len(ordered):
            raise ValueError(f"split {name!r} needs {n} rows, only {len(ordered) - start} left")
        out[name] = list(ordered[start : start + n])
        start += n
    if rest is not None:
        out[rest] = list(ordered[start:])
    return out


class Decontaminator:
    """Flags training text that shares an n-word sequence with any evaluation text.

    Evaluation texts shorter than ``n`` words cannot form an n-gram, so those
    (if at least ``min_short_words`` long) are checked as normalized substrings
    instead. Otherwise a short evaluation prompt could be copied verbatim into
    training data and pass.
    """

    def __init__(self, eval_texts: Iterable[str], n: int = 8, min_short_words: int = 4):
        self.n = n
        self.grams: set[tuple[str, ...]] = set()
        self.short: list[str] = []
        self.n_eval = 0
        for t in eval_texts:
            self.n_eval += 1
            g = word_ngrams(t, n)
            if g:
                self.grams |= g
            elif len(norm_text(t).split()) >= min_short_words:
                self.short.append(norm_text(t))

    def reason(self, text: str) -> str | None:
        if word_ngrams(text, self.n) & self.grams:
            return f"eval_overlap:{self.n}gram"
        normed = norm_text(text)
        if any(s in normed for s in self.short):
            return "eval_overlap:short_prompt_substring"
        return None


def curate_sft(
    rows: Sequence[dict],
    *,
    source_uri: str,
    decontaminator: Decontaminator,
    min_devanagari: float = 0.6,
    minhash_threshold: float = 0.85,
    minhash_ngram: int = 5,
) -> tuple[list[dict], Any, dict]:
    """Curate Hindi instruction pairs with CuratorKIT. Rows need ``prompt`` and ``response``.

    Steps, in order: reader (field mapping) → TextCleaner (NFC first, so dedup
    compares normalized text) → SchemaGate → DevanagariGate → exact dedup →
    MinHash near-dup (5-character shingles: Devanagari vowel signs are separate
    code points, so 3-character shingles over-match) → decontamination.

    Returns ``(kept_rows, pipeline_result, step_config)``.
    """
    from curatorkit.connectors.base import BaseConnector
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.interfaces import BaseGate
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.pipeline import Pipeline
    from curatorkit.schema import RejectedSample

    class ListReader(BaseConnector):
        def __init__(self, items: Sequence[dict], **kw: Any) -> None:
            self.items = list(items)
            super().__init__(**kw)

        def _iter_rows(self):
            yield from enumerate(self.items, start=1)

    class ReasonGate(BaseGate):
        def reason(self, sample) -> str | None:  # pragma: no cover - abstract
            raise NotImplementedError

        def run(self, samples):
            passed, rejected = [], []
            for s in samples:
                why = self.reason(s)
                if why is None:
                    passed.append(s)
                else:
                    rejected.append(RejectedSample(**s.model_dump(), rejection_reason=why,
                                                   rejecting_step=type(self).__name__))
            return passed, rejected

    class DevanagariGate(ReasonGate):
        def reason(self, s):
            r = devanagari_ratio(f"{s.instruction} {s.output}")
            return None if r >= min_devanagari else "script:devanagari_below_threshold"

    class DecontaminationGate(ReasonGate):
        def reason(self, s):
            return decontaminator.reason(f"{s.instruction}\n{s.output}")

    steps = [
        ListReader(rows, source_uri=source_uri, format="alpaca",
                   field_mapping={"prompt": "instruction", "response": "output"}),
        TextCleaner(),
        SchemaGate(min_tokens=3, max_tokens=1536),
        DevanagariGate(),
        ExactDeduplicator(),
        MinHashDeduplicator(threshold=minhash_threshold, ngram=minhash_ngram),
        DecontaminationGate(),
    ]
    result = Pipeline(steps).run()
    kept = []
    for s in result.passed:
        meta = dict(s.metadata or {})
        kept.append({
            "id": meta.get("id") or stable_id("sft", s.instruction, s.output),
            "prompt": s.instruction,
            "response": s.output,
        })
    config = {
        "min_devanagari": min_devanagari,
        "schema_min_tokens": 3,
        "schema_max_tokens": 1536,
        "minhash_threshold": minhash_threshold,
        "minhash_ngram": minhash_ngram,
        "decontam_ngram": decontaminator.n,
        "decontam_eval_texts": decontaminator.n_eval,
    }
    return kept, result, config


def to_jsonl(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
