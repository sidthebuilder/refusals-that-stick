# Refusals That Stick

**Do weight-space safety repairs survive deployment quantization?**

Lexsi AI Safety and Alignment Hackathon · Track 2b (AI Safety and Correction) ·
`CohereLabs/tiny-aya-fire` (3.35B) · CuratorKIT + SafeTune + CircuitKIT + AuditKIT

> **Status: in progress.** The measurement tool and its test suite are done and passing. No model
> has been fine-tuned, repaired or evaluated yet, so this README contains **no results**. The
> frozen plan is in [`PREREGISTRATION.md`](./PREREGISTRATION.md); every deviation is logged in
> [`SPRINT_LOG.md`](./SPRINT_LOG.md).

## The problem

You fine-tune an aligned model for your language. Safety drifts. You repair it in the weights with
a published method — SafeTune's C-ΔΘ touches under 5% of parameters. Then you quantize to GGUF so
it runs on a phone.

**Is the repair still there?**

MANSU (arXiv:2605.15138) proved that 4-bit quantization can erase an *unlearning* edit outright: a
per-weight update smaller than one quantization bin rounds back to where it started. It explicitly
notes the repair setting does not require that guarantee. Safety repairs are also small and sparse.

A full-text search of the C-ΔΘ paper finds no mention of quantization. Neither does the card for
Lexsi's own shipped repair checkpoint, which is offered as a drop-in model "served the way you
serve it today". GitHub has no repository combining Lexsi's stack with quantization.

So this is measurable and open.

## What is built and verified

`src/refusals_that_stick/survival.py` measures, per tensor, how much of a weight edit survives a
given GGUF quantization. It runs on CPU with no GPU, no network and no API keys.

Two findings from the test suite already shaped the design, and both are pinned by tests:

**The popular quants cannot be simulated.** The `gguf` package implements quantization only for
legacy types; Q4_K, Q3_K and Q2_K are dequantize-only. So survival is measured from **real GGUF
files** built by `llama-quantize` and read back — the actual deployed artifact, not a model of it.

**Two natural metrics are badly wrong.** A magnitude ratio scored **1.41** on an edit that was
pure noise. A least-squares projection scored **1.93** on a sparse edit where **97% of edited
weights had rounded straight back**. The headline metric is therefore per-weight permanence,
bounded in [0, 1], following MANSU's Lemma 1. Details in [`PREREGISTRATION.md`](./PREREGISTRATION.md#2-the-metric-and-two-rejected-alternatives).

A third finding became a pre-registered hypothesis: erasure is driven by edit **sparsity**, not
edit size. A circuit-restricted repair should therefore be more fragile than a whole-model one.

## Reproduce the tool

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m pytest tests -q      # 25 tests, CPU, a few seconds
```

```python
from refusals_that_stick.survival import tensor_survival, from_gguf_pair

# Simulated, legacy quant types only
r = tensor_survival(before_weights, after_weights, "Q4_0")
print(r.directional_permanence)   # 1.0 preserved, 0.0 erased

# Real files, any quant type including k-quants
for t in from_gguf_pair("drifted-Q4_K_M.gguf", "repaired-Q4_K_M.gguf"):
    print(t.name, t.permanence)
```

## Layout

| Path | Contents |
|---|---|
| `PREREGISTRATION.md` | Hypotheses, metric, decision rules — frozen before any result |
| `SPRINT_LOG.md` | Every deviation, with evidence |
| `src/refusals_that_stick/survival.py` | The survival measurement |
| `tests/` | 25 CPU tests, including the two rejected-metric findings |

## Licensing and prior work

`CohereLabs/tiny-aya-fire` is **CC-BY-NC**, so every derived artifact here is non-commercial,
including the ROI comparison.

Lexsi's CuratorKIT, SafeTune, CircuitKIT and AuditKIT are source-available under LSAL and are
**referenced, never redistributed**; they are cloned locally into a gitignored `_vendor/`.

Nothing in this repository predates the hackathon. The measurement tool, tests, pre-registration
and plan were all written for it.

## Author

Shashank Kumar — solo entry.
