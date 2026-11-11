"""Count the words in the submission abstract against the 500-word limit.

The limit applies to the abstract section only, so the heading, the links and the
reproducibility note are excluded. Markdown emphasis and link syntax are stripped
before counting, so formatting cannot inflate or deflate the number.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LIMIT = 500
ROOT = Path(__file__).resolve().parents[1]


def abstract_words(text: str) -> list[str]:
    body = text.split("## Abstract", 1)[1]
    body = body.split("\n## ", 1)[0]
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)   # links -> their text
    body = re.sub(r"[*_`#>|]", " ", body)
    return [w for w in body.split() if re.search(r"\w", w)]


def main() -> int:
    path = ROOT / "SUBMISSION.md"
    words = abstract_words(path.read_text(encoding="utf-8"))
    n = len(words)
    print(f"abstract: {n} words (limit {LIMIT})")
    if n > LIMIT:
        print(f"OVER by {n - LIMIT}")
        return 1
    print(f"{LIMIT - n} words spare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
