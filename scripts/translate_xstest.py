"""Make the Hindi XSTest split files from the English ones.

Usage (from the repository root, after the data build has written data/eval):

    python scripts/translate_xstest.py

Reads data/eval/xstest_en_{dev,test}.jsonl and writes data/eval/xstest_hi_{dev,test}.jsonl
plus data/eval/xstest_hi_translation_report.json. Needs ADAPTION_API_KEY and
ADAPTION_BASE_URL in the environment or in .env. Prints counts only, never text
of the key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refusals_that_stick.translate import translate_xstest  # noqa: E402

EVAL = ROOT / "data" / "eval"


def main() -> int:
    reports = {}
    for split in ("dev", "test"):
        src = EVAL / f"xstest_en_{split}.jsonl"
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        out, report = translate_xstest(rows)
        dst = EVAL / f"xstest_hi_{split}.jsonl"
        dst.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                               for r in out), encoding="utf-8", newline="\n")
        reports[split] = report
        print(f"{split}: {report['n']} translated, chrF median {report['chrf_median']}, "
              f"min {report['chrf_min']}, flagged {report['n_flagged']}")
    (EVAL / "xstest_hi_translation_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
