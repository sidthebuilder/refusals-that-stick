"""Run a GGUF behind llama.cpp's ``llama-server`` and query it concurrently.

Every model in every cell is served the same way: same binary (pinned commit),
same flags, same chat template (the one embedded in the GGUF, rendered by
llama.cpp's Jinja engine via ``--jinja``), greedy decoding. Without ``--jinja``,
llama.cpp falls back to built-in template heuristics, which would drop the
safety preamble that Fire's own template injects.

Requests run in a thread pool sized to the server's parallel slots; the server
batches them continuously. Batched decoding is not bit-reproducible across runs
(floating-point summation order depends on batch composition), so outputs can
differ slightly between runs of the same cell. That noise is the same for every
cell and is absorbed by the paired, per-prompt statistics.

Standard library plus ``requests`` only.
"""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

LLAMA_BIN = Path("/opt/llama.cpp/build/bin")

_local = threading.local()


def _session():
    import requests

    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        _local.session = s
    return s


class LlamaServer:
    """Context manager around one ``llama-server`` process."""

    def __init__(
        self,
        gguf: str | Path,
        *,
        port: int = 8080,
        n_parallel: int = 32,
        ctx_per_slot: int = 2048,
        log_path: str | Path | None = None,
        extra_args: Sequence[str] = (),
        startup_timeout_s: float = 900.0,
    ) -> None:
        self.gguf = str(gguf)
        self.port = port
        self.n_parallel = n_parallel
        self.ctx_per_slot = ctx_per_slot
        self.base = f"http://127.0.0.1:{port}"
        # The live log is always written to local disk. ``log_path`` (often on
        # the Modal volume) receives a copy when the server stops: a file held
        # open on the volume blocks volume.reload() in the next call.
        self.live_log = Path(f"/tmp/llama-server-{port}.log")
        self.log_path = Path(log_path) if log_path else None
        self.extra_args = list(extra_args)
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self._log_fh = None
        self.load_seconds: float | None = None

    @property
    def command(self) -> list[str]:
        return [
            str(LLAMA_BIN / "llama-server"),
            "-m", self.gguf,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", "999",
            "-c", str(self.n_parallel * self.ctx_per_slot),
            "-np", str(self.n_parallel),
            "--jinja",
            "--seed", "0",
            *self.extra_args,
        ]

    def log_tail(self, n: int = 4000) -> str:
        try:
            return self.live_log.read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""

    def __enter__(self) -> "LlamaServer":
        self._log_fh = self.live_log.open("w", encoding="utf-8")
        t0 = time.time()
        self.proc = subprocess.Popen(self.command, stdout=self._log_fh, stderr=subprocess.STDOUT)
        while True:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.__exit__(None, None, None)
                raise RuntimeError(f"llama-server exited with {code}:\n{self.log_tail()}")
            try:
                r = _session().get(f"{self.base}/health", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001 - not listening yet
                pass
            if time.time() - t0 > self.startup_timeout_s:
                self.__exit__(None, None, None)
                raise TimeoutError(f"llama-server not healthy after {self.startup_timeout_s}s:\n"
                                   f"{self.log_tail()}")
            time.sleep(1.0)
        self.load_seconds = round(time.time() - t0, 1)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self.log_path.write_text(self.log_tail(200_000), encoding="utf-8")
            except OSError:
                pass  # a missing log copy must never mask the real error

    # -- HTTP -----------------------------------------------------------------

    def _post(self, path: str, body: dict, timeout: float = 600.0, retries: int = 3) -> dict:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = _session().post(f"{self.base}{path}", json=body, timeout=timeout)
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001 - retried, then re-raised with context
                last = e
                if self.proc and self.proc.poll() is not None:
                    break  # the server died; retrying cannot help
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"POST {path} failed after {retries + 1} attempts: {last}\n"
                           f"{self.log_tail(1500)}")

    def chat(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.0,
             repeat_penalty: float = 1.1, repeat_last_n: int = 64) -> dict:
        # Greedy (temperature 0) keeps cells reproducible, but this model family
        # loops under greedy with no repetition penalty, worst on short answers.
        # llama.cpp's default repeat_penalty is 1.0 (off); 1.1 is the common
        # anti-loop value and is applied only to the model under test, never to
        # the judge (see complete()).
        t0 = time.time()
        data = self._post("/v1/chat/completions", {
            "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n,
            "seed": 0, "stream": False,
        })
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return {
            "response": ((choice.get("message") or {}).get("content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_s": round(time.time() - t0, 3),
        }

    def complete(self, prompt: str, *, n_predict: int, temperature: float = 0.0) -> dict:
        data = self._post("/completion", {
            "prompt": prompt, "n_predict": n_predict, "temperature": temperature,
            "seed": 0, "cache_prompt": False, "stream": False,
        })
        return {"text": data.get("content") or "", "stop_type": data.get("stop_type"),
                "tokens_predicted": data.get("tokens_predicted")}

    def apply_template(self, messages: list[dict]) -> str:
        return self._post("/apply-template", {"messages": messages}).get("prompt", "")

    def map(self, fn: Callable[[Any], dict], items: Sequence[Any],
            workers: int | None = None) -> list[dict]:
        """Run ``fn`` over ``items`` concurrently, preserving order."""
        with ThreadPoolExecutor(max_workers=workers or self.n_parallel) as pool:
            return list(pool.map(fn, items))


def perplexity(gguf: str | Path, text_file: str | Path, *, ctx: int = 512,
               timeout_s: float = 1800.0) -> dict:
    """``llama-perplexity`` on a text file; returns the final estimate and its error."""
    import re

    r = subprocess.run(
        [str(LLAMA_BIN / "llama-perplexity"), "-m", str(gguf), "-f", str(text_file),
         "-c", str(ctx), "-ngl", "999"],
        capture_output=True, text=True, timeout=timeout_s,
    )
    text = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if r.returncode != 0 or not m:
        return {"ok": False, "returncode": r.returncode, "tail": text[-2000:]}
    chunks = re.search(r"calculating perplexity over (\d+) chunks", text)
    return {"ok": True, "ppl": float(m.group(1)), "ppl_stderr": float(m.group(2)),
            "ctx": ctx, "chunks": int(chunks.group(1)) if chunks else None}
