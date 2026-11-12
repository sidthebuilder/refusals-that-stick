"""Figures for the README and the write-up, built only from committed files in results/.

    python scripts/make_figures.py          # writes docs/figures/*.png

Five figures:

1. ``overview.png``       the three headline numbers, for the top of the README
2. ``repair_frontier.png`` all 13 repair configurations against the pre-registered gate
3. ``weight_survival.png`` what happens to every edited weight at each GGUF precision
4. ``behaviour_test.png``  benefit lost at Q4_K_M on the held-out test split, with 95% intervals
5. ``power_curve.png``     how many prompts a behavioural test needs to see that loss

Needs matplotlib (not a dependency of the package itself).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
SPARSE = "#c2410c"
DENSE = "#0f766e"
ERASED = "#d1d5db"
AGAINST_S = "#fdba74"
AGAINST_D = "#99f6e4"
FAMILY = {"safemerge": SPARSE, "resta": DENSE, "lox-r8": "#6d28d9", "lox-r6": "#a78bfa",
          "ctheta": "#2563eb"}
FAMILY_NAME = {"safemerge": "SafeMerge", "resta": "RESTA", "lox-r8": "LoX (rank 8)",
               "lox-r6": "LoX (rank 6)", "ctheta": "C-ΔΘ"}
PRECISIONS = ("Q8_0", "Q6_K", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K")

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlesize": 13,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
})


def load(name: str):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.965, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.905, subtitle, fontsize=11, color=MUTED, va="top")


def footer(fig, text: str) -> None:
    fig.text(0.04, 0.02, text, fontsize=8.5, color=MUTED, va="bottom")


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
def fig_overview(surv: list[dict], qc: dict) -> Path:
    s = next(r for r in surv if r["arm"] == "sparse" and r["precision"] == "Q4_K_M")
    d = next(r for r in surv if r["arm"] == "dense" and r["precision"] == "Q4_K_M")
    sh = next(r for r in qc["results"] if r["arm"] == "sparse" and r["lang"] == "hi")
    tiles = [
        ("290 / 290", "tensors bit-identical to Cohere's\npublished GGUF files", INK),
        (f"{s['direction_among_moved']:.1%}",
         f"of the sparse repair's moved weights\ngo the intended way at Q4_K_M\n"
         f"(dense repair: {d['direction_among_moved']:.1%}; coin flip: 50%)", SPARSE),
        (f"≈{round(sh['n_for_80pct_power'], -2):,.0f}",
         "red-team prompts per language needed\nto detect the behavioural loss\n"
         f"(held-out test split here: {sh['n']})", DENSE),
    ]
    fig = plt.figure(figsize=(12, 4.2))
    header(fig, "Refusals That Stick",
           "Do weight-space safety repairs survive GGUF quantization?  "
           "Tiny Aya Fire (3.35B), Hindi and English.")
    for i, (big, small, colour) in enumerate(tiles):
        x0 = 0.04 + i * 0.315
        ax = fig.add_axes([x0, 0.12, 0.29, 0.66])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="#f9fafb", edgecolor=GRID, linewidth=1.2,
                                    transform=ax.transAxes))
        ax.add_patch(Rectangle((0, 0.93), 1, 0.07, facecolor=colour, transform=ax.transAxes,
                               clip_on=False))
        ax.text(0.07, 0.70, big, fontsize=30, fontweight="bold", color=colour,
                transform=ax.transAxes, va="center")
        ax.text(0.07, 0.36, small, fontsize=10.5, color=INK, transform=ax.transAxes,
                va="center", linespacing=1.4)
    return save(fig, "overview.png")


# --------------------------------------------------------------------------------------
def fig_frontier(sel: dict) -> Path:
    fig = plt.figure(figsize=(11, 6.6))
    header(fig, "No repair configuration reaches the gate",
           "13 configurations of four SafeTune repair methods on the drifted model, at BF16. "
           "The shaded box is the pre-registered target.")
    ax = fig.add_axes([0.08, 0.13, 0.62, 0.68])
    ax.add_patch(Rectangle((0, 50), 5, 120, facecolor="#dcfce7", edgecolor="#16a34a",
                           linewidth=1, linestyle="--", zorder=0))
    ax.text(0.6, 157, "Gate:\n≥50% restored,\n≤5 pt MMLU lost", fontsize=9, color="#166534",
            va="top")
    for r in sel["rows"]:
        fam = r["family"]
        loss = -100 * min(r["mmlu_vs_drift"]["en"], r["mmlu_vs_drift"]["hi"])
        rest = 100 * r["restoration_hi"]
        over = not r["checks"]["overrefusal"]
        ax.scatter([loss], [rest], s=90, zorder=3,
                   facecolor="white" if over else FAMILY[fam], edgecolor=FAMILY[fam],
                   linewidth=2)
        name = r["model"]
        if name in ("safemerge-dose3-s1", "resta-dose3-s0.25"):
            tag = "sparse arm\n(SafeMerge α1.0)" if "safemerge" in name else "dense arm\n(RESTA α0.25)"
            ax.annotate(tag, (loss, rest), xytext=(14, -6), textcoords="offset points",
                        fontsize=9.5, color=FAMILY[fam], fontweight="bold")
    ax.set_xlim(-2, 68)
    ax.set_ylim(-15, 165)
    ax.set_xlabel("MMLU points lost against the drifted model (worse of Hindi and English)")
    ax.set_ylabel("Share of the Hindi refusal drop restored (%)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(67, 102, "back to Fire's level", fontsize=8.5, color=MUTED, ha="right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Legend from proxies, so a method's colour is never confused with the hollow
    # (over-refusal) marker style.
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                          markerfacecolor=FAMILY[f], markeredgecolor=FAMILY[f], label=FAMILY_NAME[f])
               for f in ("safemerge", "resta", "lox-r8", "lox-r6", "ctheta")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=False, title="Method", fontsize=10)
    leg.get_title().set_fontweight("bold")
    fig.text(0.725, 0.47, "Hollow marker: also fails\nthe over-refusal check\n"
             "(benign prompts refused\n>10 pt more than Fire).", fontsize=9.5, color=MUTED,
             va="top", linespacing=1.4)
    footer(fig, "Source: results/repair_selection.json. Dev split, 150 red-team prompts per "
                "language. The two labelled points are the repairs carried into the "
                "quantization study.")
    return save(fig, "repair_frontier.png")


# --------------------------------------------------------------------------------------
def fig_survival(surv: list[dict]) -> Path:
    fig = plt.figure(figsize=(12, 6.2))
    header(fig, "What quantization does to every edited weight",
           "Each edited weight ends up erased, moved the way the repair intended, or moved "
           "against it. Measured on real GGUF files.")
    panels = [("sparse", "Sparse repair: SafeMerge α1.0\n274M weights in 15 of 290 tensors",
               SPARSE, AGAINST_S),
              ("dense", "Dense repair: RESTA α0.25\n3.26B weights in all 290 tensors",
               DENSE, AGAINST_D)]
    for i, (arm, title, c_int, c_ag) in enumerate(panels):
        ax = fig.add_axes([0.09 + i * 0.45, 0.17, 0.34, 0.62])
        rows = {r["precision"]: r for r in surv if r["arm"] == arm}
        ys = list(range(len(PRECISIONS)))[::-1]
        for y, p in zip(ys, PRECISIONS):
            r = rows[p]
            intended = r["directional_permanence"]
            against = r["moved_against_fraction"]
            erased = r["erased_fraction"]
            ax.barh(y, intended, color=c_int, height=0.62)
            ax.barh(y, against, left=intended, color=c_ag, height=0.62)
            ax.barh(y, erased, left=intended + against, color=ERASED, height=0.62)
            bold = "bold" if p == "Q4_K_M" else "normal"
            ax.text(1.03, y, f"{r['direction_among_moved']:.0%}", va="center", fontsize=10.5,
                    color=c_int, fontweight=bold)
        ax.set_yticks(ys)
        ax.set_yticklabels(PRECISIONS)
        for lab in ax.get_yticklabels():
            if lab.get_text() == "Q4_K_M":
                lab.set_fontweight("bold")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, loc="left", fontsize=11.5, color=c_int, fontweight="bold")
        ax.text(1.03, len(PRECISIONS) - 0.35, "intended\namong moved", fontsize=8.5,
                color=MUTED, va="bottom")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    # Legend: each state shown in both panels' colours.
    lx = 0.09
    for label, colours in (("moved the intended way", (SPARSE, DENSE)),
                           ("moved against the repair", (AGAINST_S, AGAINST_D)),
                           ("erased (rounded back)", (ERASED, ERASED))):
        for j, colour in enumerate(colours):
            fig.patches.append(Rectangle((lx + j * 0.013, 0.075), 0.012, 0.03,
                                         transform=fig.transFigure, facecolor=colour,
                                         edgecolor="none"))
        fig.text(lx + 0.032, 0.09, label, fontsize=9.5, color=INK, va="center")
        lx += 0.24
    footer(fig, "Source: results/survival_analysis.json, a census of every edited weight. "
                "Right-hand number: intended ÷ (intended + against); a coin flip gives 50%.")
    return save(fig, "weight_survival.png")


# --------------------------------------------------------------------------------------
def fig_behaviour(qc: dict) -> Path:
    fig = plt.figure(figsize=(11, 5.2))
    header(fig, "On held-out prompts, the behavioural loss is within noise",
           "Repair benefit lost at Q4_K_M, per prompt: (repair − drift) at BF16 minus "
           "(repair − drift) at Q4_K_M. 663 prompts per language.")
    ax = fig.add_axes([0.24, 0.17, 0.66, 0.62])
    order = [("sparse", "hi"), ("sparse", "en"), ("dense", "hi"), ("dense", "en")]
    names = {"sparse": "Sparse repair", "dense": "Dense repair"}
    langs = {"hi": "Hindi", "en": "English"}
    for y, (arm, lang) in zip(range(len(order))[::-1], order):
        r = next(x for x in qc["results"] if x["arm"] == arm and x["lang"] == lang)
        c = SPARSE if arm == "sparse" else DENSE
        lo, hi = 100 * r["ci95"][0], 100 * r["ci95"][1]
        m = 100 * r["benefit_lost"]
        ax.plot([lo, hi], [y, y], color=c, linewidth=3, solid_capstyle="round")
        ax.scatter([m], [y], s=110, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(hi + 0.4, y, f"{m:+.1f} pt  [{lo:+.1f}, {hi:+.1f}]", va="center",
                fontsize=10, color=INK)
        ax.text(-7.6, y, f"{names[arm]}, {langs[lang]}", va="center", ha="right",
                fontsize=10.5, color=INK)
        ax.text(-7.6, y - 0.28, f"benefit {100 * r['benefit_bf16']:+.1f} pt at BF16",
                va="center", ha="right", fontsize=8.5, color=MUTED)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlim(-7, 11)
    ax.set_ylim(-0.75, 3.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Refusal-rate points lost to quantization (positive = repair weakened)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    footer(fig, "Source: results/quant_cost_test.json. Dot: mean; bar: 95% percentile "
                "bootstrap interval over prompts (10,000 draws, seed 0). No interval "
                "excludes zero.")
    return save(fig, "behaviour_test.png")


# --------------------------------------------------------------------------------------
def fig_power(qc: dict) -> Path:
    r = next(x for x in qc["results"] if x["arm"] == "sparse" and x["lang"] == "hi")
    se, n0, loss = r["se"], r["n"], 100 * r["benefit_lost"]
    fig = plt.figure(figsize=(11, 5.6))
    header(fig, "Why a behavioural test cannot see the weight-level loss",
           "Smallest loss of refusal a paired test can detect (80% power, 5% two-sided), "
           "for the sparse repair in Hindi.")
    ax = fig.add_axes([0.09, 0.15, 0.84, 0.64])
    ns = [100 * 1.06 ** k for k in range(0, 95)]
    mde = [100 * 2.8 * se * (n0 / n) ** 0.5 for n in ns]
    ax.plot(ns, mde, color=SPARSE, linewidth=2.5)
    ax.axhline(loss, color=INK, linewidth=1, linestyle="--")
    ax.text(105, loss + 0.25, f"observed loss: {loss:.1f} pt", fontsize=9.5, color=INK)
    marks = [(150, "dev split\n150"), (n0, f"test split\n{n0}"),
             (913, "every Hindi Aya\nred-team prompt\n913"),
             (r["n_for_80pct_power"], f"needed\n≈{round(r['n_for_80pct_power'], -2):,.0f}")]
    for n, label in marks:
        y = 100 * 2.8 * se * (n0 / n) ** 0.5
        colour = DENSE if label.startswith("needed") else MUTED
        ax.scatter([n], [y], s=60, color=colour, zorder=3)
        ax.annotate(label, (n, y), xytext=(8, 10), textcoords="offset points", fontsize=9,
                    color=colour if label.startswith("needed") else INK, linespacing=1.2)
    ax.set_xscale("log")
    ax.set_xlim(100, 25000)
    ax.set_ylim(0, 13)
    ax.set_xticks([100, 300, 1000, 3000, 10000])
    ax.set_xticklabels(["100", "300", "1,000", "3,000", "10,000"])
    ax.set_xlabel("Red-team prompts per language")
    ax.set_ylabel("Detectable loss (refusal-rate points)")
    ax.grid(color=GRID, which="both", linewidth=0.7)
    ax.set_axisbelow(True)
    footer(fig, "Source: results/quant_cost_test.json. Standard error from the bootstrap "
                "interval, scaled as 1/√n. Detectable loss = 2.8 × standard error.")
    return save(fig, "power_curve.png")


def main() -> int:
    surv = load("survival_analysis.json")["rows"]
    qc = load("quant_cost_test.json")
    sel = load("repair_selection.json")
    for p in (fig_overview(surv, qc), fig_frontier(sel), fig_survival(surv),
              fig_behaviour(qc), fig_power(qc)):
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
