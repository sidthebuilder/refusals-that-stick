"""Paired statistics for per-prompt binary outcomes, as pre-registered.

Every comparison in the study is paired: the same prompts, scored in two
cells. For binary outcomes (unsafe or not, refused or not) the pre-registered
test is McNemar's, computed exactly from the discordant pairs. Multiple tests
within a hypothesis family are Holm-corrected. Rates are reported with Wilson
score intervals.

Standard library only, so the analysis runs anywhere.
"""

from __future__ import annotations

import math
from typing import Sequence  # noqa: F401 - used in annotations below

__all__ = ["binom_two_sided", "mcnemar_exact", "wilson", "holm", "paired_binary",
           "ranks", "spearman"]


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value: total probability of outcomes no likelier than k."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    # Relative tolerance guards ties against floating-point noise.
    return min(1.0, sum(q for q in pmf if q <= obs * (1 + 1e-9)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value from discordant counts b (A yes, B no) and c (A no, B yes)."""
    return binom_two_sided(min(b, c), b + c)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    # The bounds are exactly 0 and 1 at the extremes; set them so float
    # rounding cannot print a lower bound of 1.7e-18 in a results table.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_binary(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Compare two cells on the prompts both scored. ``a``/``b`` map prompt id to outcome."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    ka = sum(1 for i in ids if a[i])
    kb = sum(1 for i in ids if b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    a_only = sum(1 for i in ids if a[i] and not b[i])
    return {
        "n": n,
        "rate_a": ka / n if n else None,
        "rate_b": kb / n if n else None,
        "ci_a": wilson(ka, n),
        "ci_b": wilson(kb, n),
        "delta": (kb - ka) / n if n else None,
        "discordant_a_only": a_only,
        "discordant_b_only": b_only,
        "p_mcnemar": mcnemar_exact(a_only, b_only),
    }


def ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, with tied values sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation. ``None`` when either side has no variation."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    dx = [a - mx for a in rx]
    dy = [b - my for b in ry]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return None if den == 0 else num / den
