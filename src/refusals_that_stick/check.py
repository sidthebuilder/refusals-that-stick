"""Check whether a weight edit survives GGUF quantization, and print the fix.

Give it two lossless GGUF files of one model, before and after an edit (a safety
repair, or any other weight edit), and the precisions you plan to ship. For each
precision it reports, over the weights the edit changed:

* **erased**: the share that rounds back to its pre-edit value;
* **intended among moved**: of the weights whose value still differs, the share that
  moved the way the edit intended. A coin flip gives 50%;
* **directional survival**: the share of all edited weights that moved the intended
  way.

When Q8_0 is measured too, it also gives, for every lower precision, the exact
result of keeping the edited tensors at Q8_0 inside that file, the size cost, and the
``llama-quantize`` arguments that build it. Q8_0 quantizes each tensor on its own, so
the protected file needs no extra run to predict (see ``protection.py``);
``--verify`` builds it anyway and measures it.

    python -m refusals_that_stick.check BEFORE.gguf AFTER.gguf \\
        --precisions Q8_0,Q4_K_M --llama-quantize path/to/llama-quantize

or, with files that are already quantized (both files of a pair the same way):

    python -m refusals_that_stick.check BEFORE.gguf AFTER.gguf \\
        --quantized Q8_0=before-q8_0.gguf,after-q8_0.gguf \\
        --quantized Q4_K_M=before-q4_k_m.gguf,after-q4_k_m.gguf

CPU only. Needs ``gguf`` and ``numpy``; quantizing needs a ``llama-quantize`` binary.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from .protection import protected_survival
from .survival_map import survival_map

__all__ = ["check_repair", "format_report", "tensor_type_args", "main"]

LOSSLESS = ("F32", "F16", "BF16")
# Above this many tensors, the printed command lists only the first few names; the
# full argument list is always in the JSON report.
PRINT_LIMIT = 40


def _tensor_types(path: str | Path) -> list[dict[str, Any]]:
    from gguf.constants import GGMLQuantizationType
    from gguf.gguf_reader import GGUFReader

    return [{"name": str(t.name), "qtype": GGMLQuantizationType(t.tensor_type).name,
             "n_params": int(t.n_elements), "n_bytes": int(t.n_bytes)}
            for t in GGUFReader(str(path), "r").tensors]


def tensor_type_args(names: Sequence[str], qtype: str = "q8_0") -> list[str]:
    """``llama-quantize`` arguments that set exactly these tensors to ``qtype``."""
    args: list[str] = []
    for name in sorted(names):
        args += ["--tensor-type", f"^{re.escape(name)}$={qtype}"]
    return args


def _payload(type_rows: Sequence[dict], protect: Sequence[str] = ()) -> int:
    """Tensor bytes of a file as built, or with ``protect`` re-typed to Q8_0."""
    protect = set(protect)
    total = 0
    for r in type_rows:
        if r["name"] in protect and r["qtype"] not in LOSSLESS + ("Q8_0",):
            total += r["n_params"] * 34 // 32  # Q8_0: 34 bytes per block of 32
        else:
            total += r["n_bytes"]
    return total


def _quantize(binary: str | Path, src: str | Path, dst: Path, qtype: str,
              extra_args: Sequence[str] = (), threads: int | None = None) -> dict:
    tmp = dst.with_suffix(".partial")
    cmd = [str(binary), *extra_args, str(src), str(tmp), qtype]
    if threads:
        cmd.append(str(threads))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"llama-quantize {qtype} failed:\n{(r.stderr or r.stdout or '')[-3000:]}")
    tmp.replace(dst)
    return {"seconds": round(time.time() - t0, 1), "bytes": dst.stat().st_size}


def _summary(permanence: float | None, directional: float | None,
             erased: float | None) -> dict[str, float | None]:
    return {"erased": erased, "directional_survival": directional,
            "intended_among_moved": (directional / permanence) if permanence else None}


def check_repair(before: str | Path, after: str | Path, *,
                 precisions: Sequence[str] = ("Q8_0", "Q4_K_M"),
                 quantized: dict[str, tuple[str, str]] | None = None,
                 llama_quantize: str | Path | None = None,
                 workdir: str | Path | None = None,
                 verify: bool = False,
                 threads: int | None = None) -> dict[str, Any]:
    """Measure the edit ``after - before`` at each precision; see the module docstring."""
    quantized = {k.upper(): v for k, v in (quantized or {}).items()}
    wanted = list(dict.fromkeys([p.strip().upper() for p in precisions] + list(quantized)))
    wanted.sort(key=lambda p: p != "Q8_0")  # stable: Q8_0 first, the rest as given
    for path in (before, after):
        lossy = sorted({r["qtype"] for r in _tensor_types(path)} - set(LOSSLESS))
        if lossy:
            raise ValueError(f"{path} holds quantized tensors ({', '.join(lossy)}); "
                             "before and after must be lossless (F32, F16 or BF16)")
    if verify and not llama_quantize:
        raise ValueError("--verify builds the protected file and needs --llama-quantize")
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="rts-check-"))
    work.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"before": str(before), "after": str(after), "precisions": {},
                              "protection": {}, "built": {}}
    maps: dict[str, tuple[dict, Path]] = {}
    for p in wanted:
        if p in quantized:
            bq, aq = (Path(x) for x in quantized[p])
        else:
            if not llama_quantize:
                raise ValueError(f"no files given for {p}, and no --llama-quantize to build them")
            bq, aq = work / f"before-{p}.gguf", work / f"after-{p}.gguf"
            for src, dst in ((before, bq), (after, aq)):
                if not dst.exists():
                    report["built"][dst.name] = _quantize(llama_quantize, src, dst, p,
                                                          threads=threads)
        m = survival_map(str(before), str(after), str(bq), str(aq), qtype_label=p)
        maps[p] = (m, aq)
        report["precisions"][p] = {
            **_summary(m["permanence"], m["directional_permanence"], m["erased_fraction"]),
            "file": str(aq),
            "tensors": [{k: r.get(k) for k in ("name", "qtype", "n_edited", "erased_fraction",
                                               "directional_permanence", "permanence")}
                        for r in m["tensors"] if r.get("n_edited", 0) > 0],
        }
    any_map = next(iter(maps.values()))[0]
    edited = [r["name"] for r in any_map["tensors"] if r.get("n_edited", 0) > 0]
    report["edit"] = {"n_tensors": any_map["n_tensors"], "n_tensors_edited": len(edited),
                      "n_weights_edited": any_map["n_weights_edited"],
                      "share_of_weights_edited": any_map["share_of_weights_edited"],
                      "tensors": edited}
    if not edited or "Q8_0" not in maps:
        return report

    q8_map, q8_file = maps["Q8_0"]
    q8_payload = _payload(_tensor_types(q8_file))
    for p, (m, aq) in maps.items():
        if p == "Q8_0":
            continue
        types = _tensor_types(aq)
        tmap = {r["name"]: r["qtype"] for r in types}
        protect = [n for n in edited if tmap.get(n) not in LOSSLESS + ("Q8_0",)]
        if not protect:
            continue
        ps = protected_survival(m["tensors"], q8_map["tensors"], protect)
        base_b, new_b = _payload(types), _payload(types, protect)
        entry: dict[str, Any] = {
            "protect": protect,
            "computed": _summary(ps["permanence"], ps["directional_permanence"],
                                 ps["erased_fraction"]),
            "payload_bytes": base_b, "protected_payload_bytes": new_b,
            "size_increase": new_b / base_b - 1,
            "whole_model_q8_0_size_increase": q8_payload / base_b - 1,
            "llama_quantize_args": tensor_type_args(protect),
        }
        if verify:
            vb, va = work / f"before-{p}-protected.gguf", work / f"after-{p}-protected.gguf"
            for src, dst in ((before, vb), (after, va)):
                report["built"][dst.name] = _quantize(llama_quantize, src, dst, p,
                                                      entry["llama_quantize_args"], threads)
            mv = survival_map(str(before), str(after), str(vb), str(va),
                              qtype_label=f"{p}+edited@Q8_0")
            built = {r["name"]: r["qtype"] for r in _tensor_types(va)}
            entry["verified"] = {
                **_summary(mv["permanence"], mv["directional_permanence"],
                           mv["erased_fraction"]),
                "file": str(va), "file_bytes": va.stat().st_size,
                "plain_file_bytes": aq.stat().st_size,
                "protected_not_at_q8_0": sorted(n for n in protect if built.get(n) != "Q8_0"),
                "other_tensors_retyped": sorted(n for n, t in built.items()
                                                if n not in protect and tmap.get(n) != t),
                "matches_computed": abs(mv["directional_permanence"]
                                        - ps["directional_permanence"]) < 1e-9,
            }
        report["protection"][p] = entry
    return report


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def format_report(report: dict[str, Any]) -> str:
    """The report as plain text, for a terminal."""
    e = report["edit"]
    out = [f"Edit: {e['n_tensors_edited']} of {e['n_tensors']} tensors changed, "
           f"{e['n_weights_edited']:,} weights ({100 * e['share_of_weights_edited']:.1f}% "
           f"of the model)."]
    if not e["n_tensors_edited"]:
        return "\n".join(out + ["Nothing to check: the two files hold the same weights."])
    out += ["", "Survival of the edit, over the edited weights:",
            f"  {'precision':10s} {'erased':>8s} {'intended among moved':>22s} "
            f"{'directional survival':>22s}"]
    for p, s in report["precisions"].items():
        out.append(f"  {p:10s} {_pct(s['erased']):>8s} {_pct(s['intended_among_moved']):>22s} "
                   f"{_pct(s['directional_survival']):>22s}")
    out.append("  Intended among moved is 50% for a coin flip.")
    if not report["protection"]:
        if "Q8_0" not in report["precisions"]:
            out += ["", "Add Q8_0 to the precisions to get the protected-file result."]
        return "\n".join(out)
    for p, pr in report["protection"].items():
        s, c = report["precisions"][p], pr["computed"]
        n = len(pr["protect"])
        out += ["", f"Keep the {n} edited tensor{'s' if n != 1 else ''} at Q8_0 inside the "
                    f"{p} file:",
                f"  intended among moved {_pct(s['intended_among_moved']).strip()} -> "
                f"{_pct(c['intended_among_moved']).strip()}, erased "
                f"{_pct(s['erased']).strip()} -> {_pct(c['erased']).strip()}",
                f"  file size +{100 * pr['size_increase']:.1f}% "
                f"(the whole model at Q8_0: +{100 * pr['whole_model_q8_0_size_increase']:.1f}%)",
                "  computed exactly from the Q8_0 and " + p + " measurements"]
        v = pr.get("verified")
        if v:
            ok = v["matches_computed"] and not v["other_tensors_retyped"] \
                and not v["protected_not_at_q8_0"]
            out.append(f"  built and measured: intended among moved "
                       f"{_pct(v['intended_among_moved']).strip()}, "
                       f"{len(v['other_tensors_retyped'])} other tensors re-typed"
                       f"{' (matches)' if ok else ' (DOES NOT MATCH; see the JSON report)'}")
        args = pr["llama_quantize_args"]
        shown = args if n <= PRINT_LIMIT else args[:2 * 3]
        lines = [f"    {shown[i]} '{shown[i + 1]}' \\" for i in range(0, len(shown), 2)]
        if n > PRINT_LIMIT:
            lines.append(f"    ... {n - 3} more, listed in the JSON report \\")
        out += ["  Build it with:", "  llama-quantize \\", *lines,
                f"    AFTER.gguf AFTER-{p}-protected.gguf {p}"]
    return "\n".join(out)


def _parse_quantized(items: Sequence[str]) -> dict[str, tuple[str, str]]:
    out = {}
    for item in items:
        try:
            p, files = item.split("=", 1)
            b, a = files.split(",", 1)
        except ValueError:
            raise SystemExit(f"--quantized expects PRECISION=BEFORE.gguf,AFTER.gguf, got {item!r}")
        out[p.strip().upper()] = (b.strip(), a.strip())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m refusals_that_stick.check",
        description="Check whether a weight edit survives GGUF quantization, and print the fix.")
    ap.add_argument("before", help="lossless GGUF (F32, F16 or BF16) before the edit")
    ap.add_argument("after", help="lossless GGUF after the edit, same model")
    ap.add_argument("--precisions", default="Q8_0,Q4_K_M",
                    help="comma-separated llama-quantize types to check (default Q8_0,Q4_K_M)")
    ap.add_argument("--quantized", action="append", default=[], metavar="P=BEFORE,AFTER",
                    help="already-quantized pair for precision P; repeatable")
    ap.add_argument("--llama-quantize", help="path to llama-quantize, to build missing files")
    ap.add_argument("--workdir", help="keep built files here (default: a temporary directory)")
    ap.add_argument("--verify", action="store_true",
                    help="also build the protected file and measure it")
    ap.add_argument("--threads", type=int, help="threads for llama-quantize")
    ap.add_argument("--json", help="write the full report to this file")
    args = ap.parse_args(argv)

    report = check_repair(args.before, args.after,
                          precisions=[p for p in args.precisions.split(",") if p.strip()],
                          quantized=_parse_quantized(args.quantized),
                          llama_quantize=args.llama_quantize, workdir=args.workdir,
                          verify=args.verify, threads=args.threads)
    print(format_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
