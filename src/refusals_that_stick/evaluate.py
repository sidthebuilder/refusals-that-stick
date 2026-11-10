"""One evaluation cell: a GGUF file, one data split, every task.

A cell runs three stages, each cached on the volume so a rerun skips finished
work and a crash loses at most the stage in flight:

1. **generate** with the model under test (``llama-server``, greedy, the GGUF's
   own chat template);
2. **judge** every red-team and XSTest response with Qwen3Guard;
3. **perplexity** on held-out Hindi text with ``llama-perplexity``.

Per-prompt records are kept, not just rates. Every comparison downstream is
paired per prompt (McNemar, paired bootstrap), and paired tests need the rows.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .judge import parse_verdict, render_moderation_prompt
from .serve import LlamaServer, perplexity
from .textcheck import degenerate_reason, parse_choice, script_profile

# task prefix -> max new tokens. 256 tokens is enough to tell compliance from
# refusal; MMLU needs room for "The answer is (B)" in either language.
MAX_TOKENS = {"redteam": 256, "xstest": 256, "mmlu": 24}
JUDGED = ("redteam", "xstest")


def task_kind(task: str) -> str:
    return task.split("_", 1)[0]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.rename(path)


def generate(server: LlamaServer, rows: list[dict], task: str) -> list[dict]:
    max_tokens = MAX_TOKENS[task_kind(task)]

    def one(row: dict) -> dict:
        out = server.chat([{"role": "user", "content": row["prompt"]}], max_tokens=max_tokens)
        rec = {k: row[k] for k in ("id", "item", "lang", "split") if k in row}
        rec.update(task=task, prompt=row["prompt"], **out)
        rec["degenerate"] = degenerate_reason(out["response"])
        rec.update(script_profile(out["response"]))
        if task_kind(task) == "mmlu":
            rec["answer"] = row["answer"]
            rec["choice"] = parse_choice(out["response"])
            rec["correct"] = rec["choice"] == row["answer"]
        return rec

    return server.map(one, rows)


def judge(server: LlamaServer, tokenizer: Any, gens: list[dict]) -> list[dict]:
    def one(g: dict) -> dict:
        text = render_moderation_prompt(tokenizer, g["prompt"], g["response"])
        out = server.complete(text, n_predict=64)
        return {"id": g["id"], **parse_verdict(out["text"]), "raw": out["text"][:300]}

    return server.map(one, gens)


def summarize(task: str, gens: list[dict], verdicts: list[dict] | None) -> dict:
    n = len(gens)
    summary: dict[str, Any] = {
        "n": n,
        "degenerate_rate": sum(1 for g in gens if g["degenerate"]) / max(1, n),
        "truncated_rate": sum(1 for g in gens if g.get("finish_reason") == "length") / max(1, n),
        "mean_completion_tokens": sum(g.get("completion_tokens") or 0 for g in gens) / max(1, n),
    }
    if task_kind(task) == "mmlu":
        summary["accuracy"] = sum(1 for g in gens if g["correct"]) / max(1, n)
        summary["unparsed_rate"] = sum(1 for g in gens if g["choice"] is None) / max(1, n)
    if verdicts is not None:
        by_id = {v["id"]: v for v in verdicts}
        vs = [by_id[g["id"]] for g in gens]
        parsed = [v for v in vs if v["parsed"]]
        m = max(1, len(parsed))
        summary.update(
            judge_parsed_rate=len(parsed) / max(1, n),
            unsafe_rate=sum(1 for v in parsed if v["safety"] == "Unsafe") / m,
            unsafe_or_controversial_rate=sum(
                1 for v in parsed if v["safety"] in ("Unsafe", "Controversial")) / m,
            refusal_rate=sum(1 for v in parsed if v["refusal"]) / m,
        )
    return summary


def run_cell(
    *,
    gguf: str,
    cell_dir: str,
    data_dir: str,
    tasks: list[str],
    judge_gguf: str,
    judge_tokenizer_dir: str,
    ppl_text: str | None,
    n_parallel: int = 32,
    server_args: list[str] | None = None,
) -> dict:
    """Generate, judge and measure perplexity for one GGUF; resumable."""
    cell = Path(cell_dir)
    cell.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    gens_by_task: dict[str, list[dict]] = {}

    # 1. Generation with the model under test.
    todo = [t for t in tasks if not (cell / f"gen_{t}.jsonl").exists()]
    if todo:
        t0 = time.time()
        with LlamaServer(gguf, n_parallel=n_parallel, log_path=cell / "server_model.log",
                         extra_args=server_args or ()) as srv:
            timings["model_load_s"] = srv.load_seconds or 0.0
            # Record exactly what the model saw, once per cell.
            sample = _read_jsonl(Path(data_dir) / f"{todo[0]}.jsonl")[0]["prompt"]
            (cell / "rendered_prompt_example.txt").write_text(
                srv.apply_template([{"role": "user", "content": sample}]), encoding="utf-8")
            for t in todo:
                rows = _read_jsonl(Path(data_dir) / f"{t}.jsonl")
                _write_jsonl(cell / f"gen_{t}.jsonl", generate(srv, rows, t))
        timings["generate_s"] = round(time.time() - t0, 1)
    for t in tasks:
        gens_by_task[t] = _read_jsonl(cell / f"gen_{t}.jsonl")

    # 2. Judge the safety-relevant tasks.
    to_judge = [t for t in tasks if task_kind(t) in JUDGED
                and not (cell / f"judge_{t}.jsonl").exists()]
    if to_judge:
        from transformers import AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(judge_tokenizer_dir)
        # The judge (Qwen3, no sliding-window attention) parsed cleanly in every
        # first cell, so its serving configuration is left unchanged.
        with LlamaServer(judge_gguf, n_parallel=n_parallel, ctx_per_slot=4096,
                         log_path=cell / "server_judge.log", port=8081) as srv:
            for t in to_judge:
                _write_jsonl(cell / f"judge_{t}.jsonl", judge(srv, tok, gens_by_task[t]))
        timings["judge_s"] = round(time.time() - t0, 1)

    # 3. Perplexity on held-out Hindi text the fine-tune never saw.
    ppl_path = cell / "perplexity.json"
    if ppl_text and not ppl_path.exists():
        t0 = time.time()
        ppl_path.write_text(json.dumps(perplexity(gguf, ppl_text), indent=2))
        timings["perplexity_s"] = round(time.time() - t0, 1)

    summary: dict[str, Any] = {"gguf": gguf, "tasks": {}}
    for t in tasks:
        vpath = cell / f"judge_{t}.jsonl"
        verdicts = _read_jsonl(vpath) if vpath.exists() else None
        summary["tasks"][t] = summarize(t, gens_by_task[t], verdicts)
    if ppl_path.exists():
        summary["perplexity"] = json.loads(ppl_path.read_text())
    summary["timings"] = timings
    summary["gguf_bytes"] = Path(gguf).stat().st_size
    (cell / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
