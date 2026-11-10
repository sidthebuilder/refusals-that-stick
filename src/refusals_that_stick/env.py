"""Load credentials from the project ``.env`` into the process environment.

Returns the variable **names** it set, never a value. Nothing here prints, logs,
hashes or caches a secret, and a malformed line is skipped without echoing it,
because the text of a line in that file may itself be a secret.

This exists so a key can reach a script without ever appearing in a command
line, a shell history, or a captured terminal transcript. On this machine the
Hugging Face CLI is blocked by Windows Application Control, which makes the
environment-variable route the only clean one anyway.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "load_env", "require", "have"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: str | Path | None = None, *, override: bool = False) -> tuple[str, ...]:
    """Read ``.env`` and set variables, returning the names set.

    Existing environment variables win unless ``override`` is true, so a value
    exported in the shell or injected by Modal is not silently replaced.
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ()

    names: list[str] = []
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
        names.append(key)
    return tuple(names)


def have(name: str) -> bool:
    """Whether a non-empty value is available, without revealing it."""
    load_env()
    return bool(os.environ.get(name, "").strip())


def require(name: str, *, hint: str = "") -> str:
    """Return a required value, or raise naming only the variable.

    The error message never contains a value, and never the contents of the
    line it came from.
    """
    load_env()
    value = os.environ.get(name, "").strip()
    if not value:
        extra = f" {hint}" if hint else ""
        raise RuntimeError(
            f"{name} is not set. Add a line `{name}=...` to {PROJECT_ROOT / '.env'} "
            f"(which is gitignored).{extra}"
        )
    return value
