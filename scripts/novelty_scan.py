"""Prior-work scan: GitHub repositories, Hugging Face models and spaces, arXiv abstracts.

Run before any model was trained, to establish that the question this project
asks — does a weight-space safety repair survive deployment quantization? — was
not already answered. The raw output is kept in ``results/novelty_scan.txt`` so
the negative results are checkable rather than asserted.

Public APIs only, no keys. GitHub's unauthenticated search is rate-limited, so
some queries in the stored run returned HTTP 403; those are logged as errors
rather than silently dropped.

Usage:
    python scripts/novelty_scan.py [--out results/novelty_scan.txt]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATOM = "{http://www.w3.org/2005/Atom}"

GITHUB_QUERIES = ["lexsi quantization", "lexsi gguf", "tiny-aya gguf", "tiny aya quantization",
                  "tiny-aya quantized safety", "lexsi compression", "lexsi tamper",
                  "aya gguf safety"]
HF_MODEL_QUERIES = ["tiny-aya-fire", "tiny-aya gguf", "tiny-aya q4", "tiny-aya awq",
                    "tiny-aya gptq", "tiny-aya safety", "tiny-aya repaired", "tiny-aya hindi"]
HF_SPACE_QUERIES = ["tiny-aya", "aya safety", "quantization safety"]
ARXIV_QUERIES = [
    'abs:quantization AND abs:"safety" AND abs:"task arithmetic"',
    'abs:quantization AND abs:realignment',
    'abs:quantization AND abs:safety AND abs:repair',
    'abs:GGUF AND abs:safety',
    'abs:quantization AND abs:safety AND abs:multilingual',
    'abs:quantization AND abs:"safety vector"',
    'abs:quantization AND abs:refusal AND abs:circuit',
    'abs:quantization AND abs:"safety patch"',
    'abs:"llama.cpp" AND abs:safety',
    'abs:quantization AND abs:"weight editing" AND abs:safety',
]


def get(url: str, accept: str = "application/json") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "novelty-scan", "Accept": accept})
    with urllib.request.urlopen(req, timeout=40) as r:  # noqa: S310 - fixed https endpoints
        return r.read().decode("utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "novelty_scan.txt")
    args = ap.parse_args()

    lines: list[str] = [f"# Prior-work scan, run {date.today().isoformat()} by "
                        f"scripts/novelty_scan.py. Raw tool output, unedited.", ""]

    def log(s: str) -> None:
        lines.append(s)

    log("== GITHUB ==")
    for q in GITHUB_QUERIES:
        try:
            data = json.loads(get("https://api.github.com/search/repositories?per_page=10&q="
                                  + urllib.parse.quote(q)))
            log(f"[{q}] total={data.get('total_count')}")
            for it in data.get("items", [])[:10]:
                log(f"   {it['full_name']} | {(it.get('description') or '')[:140]} "
                    f"| {it.get('pushed_at')}")
        except Exception as e:  # noqa: BLE001 - a failed query is a logged result, not a crash
            log(f"[{q}] ERROR {type(e).__name__}: {e}")
        time.sleep(7)  # unauthenticated GitHub search allows ~10 requests a minute

    log("== HF MODELS ==")
    for q in HF_MODEL_QUERIES:
        try:
            data = json.loads(get(
                "https://huggingface.co/api/models?limit=40&sort=lastModified&direction=-1"
                "&search=" + urllib.parse.quote(q)))
            ids = [m["id"] for m in data]
            log(f"[{q}] n={len(ids)}: " + ", ".join(ids[:40]))
        except Exception as e:  # noqa: BLE001
            log(f"[{q}] ERROR {type(e).__name__}: {e}")

    log("== HF SPACES ==")
    for q in HF_SPACE_QUERIES:
        try:
            data = json.loads(get("https://huggingface.co/api/spaces?limit=40&search="
                                  + urllib.parse.quote(q)))
            log(f"[{q}] n={len(data)}: " + ", ".join(s["id"] for s in data[:40]))
        except Exception as e:  # noqa: BLE001
            log(f"[{q}] ERROR {type(e).__name__}: {e}")

    log("== ARXIV ==")
    for q in ARXIV_QUERIES:
        try:
            root = ET.fromstring(get(
                "https://export.arxiv.org/api/query?max_results=12&sortBy=relevance"
                "&search_query=" + urllib.parse.quote(q), accept="application/atom+xml"))
            entries = root.findall(f"{ATOM}entry")
            log(f"[{q}] n={len(entries)}")
            for e in entries:
                title = " ".join((e.findtext(f"{ATOM}title") or "").split())
                aid = (e.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
                pub = (e.findtext(f"{ATOM}published") or "")[:10]
                log(f"   {aid} {pub} | {title[:150]}")
        except Exception as e:  # noqa: BLE001
            log(f"[{q}] ERROR {type(e).__name__}: {e}")
        time.sleep(3)  # arXiv asks for one request every 3 seconds

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
