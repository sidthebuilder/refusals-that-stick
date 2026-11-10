# Sprint log

Every deviation from `PREREGISTRATION.md`, with the evidence that motivated it. Append only.

---

## 10 Nov 2026 — Day 0, setup and mechanism validation

**No model has been run. No budget spent.**

### Environment facts established

| Fact | Consequence |
|---|---|
| SafeTune requires Python ≥ 3.12 (`pyproject.toml`); this machine has only 3.11 and 3.14 | SafeTune runs **only** in a Modal image, never locally. Confirms the per-toolkit environment split. |
| CuratorKIT ≥ 3.11, AuditKIT ≥ 3.10, CircuitKIT ≥ 3.10 | These can run locally on 3.11 if needed. |
| `gguf`, `numpy`, `pytest`, `modal`, `huggingface_hub` installed in `lexsi/.venv` (3.11.9) | Local work is GGUF analysis only. |

### API facts that shaped the design

**`safetune.interventions.recover.ctheta.apply_ctheta`** (read from source, not docs):

- Signature: `(target, positive, negative, circuit_info, strength=1.0, layer_subset=None, target_modules=None)`.
- Computes `delta = positive - negative` and applies `target += strength * delta`. So
  `target` = drifted, `positive` = aligned (Fire), `negative` = base.
- Only keys ending `.weight` are touched, filtered by layer index and by module-name substring.
- **Granularity is whole `(layer, module)` tensors**, not neurons. `circuit_info` contributes only
  `layer_suggestions.layer_subset` and `.target_modules`.

**`circuitkit.quick.quantize`**: ranks layers by circuit importance; circuit node names are
`A{layer}.{head}` and `MLP {layer}`; backends `quanto` and `llmcompressor`.

**Why this matters:** llama.cpp's `--tensor-type` also selects precision per tensor by regex.
C-ΔΘ's edit unit and llama.cpp's precision unit are therefore *identical*, which makes the
circuit-guided precision fix an exact intervention rather than an approximation. This was not
obvious from the documentation and is now a design anchor.

### Two corrections forced by day-0 tests

Both were caught by `pytest` on CPU, before any GPU spend.

**1. K-quants cannot be simulated.** `gguf.quants.Q4_K.quantize_blocks` raises
`NotImplementedError`; the package can dequantize k-quants but not produce them. Only legacy types
(Q4_0, Q8_0, ...) support quantization in NumPy.

*Deviation from plan:* survival is measured from **real GGUF files** built by `llama-quantize` and
read back with `GGUFReader` + `dequantize`, rather than simulated. This is a strict improvement:
it measures the actual deployed artifact, so no simulation-fidelity argument is needed. NumPy
round-tripping is retained for legacy types as a fast pre-check and for unit tests.

**2. The headline metric was wrong, twice.**

| Candidate | Observed failure |
|---|---|
| `Σ|realized| / Σ|intended|` | **1.41** on a sub-bin Q4_0 edit with sign agreement at chance — noise scored as better than perfect |
| `⟨realized, intended⟩ / ⟨intended, intended⟩` | **1.93** on a sparse sub-bin Q8_0 edit where **97%** of edited weights had rounded back; the 3% that moved jumped a full bin and dominated the dot product |

*Deviation:* the headline is now per-weight `permanence` and `directional_permanence`, bounded in
[0, 1], matching MANSU's Lemma 1. The rejected measures are kept as diagnostics and are documented
as unquotable alone. Both failures are locked into the test suite so they cannot silently return.

### New hypothesis H0

Day-0 tests showed erasure depends on **edit sparsity**, not merely edit size:

- *Sparse* sub-bin edit: directional permanence < 0.25 on Q4_0 and Q8_0 — genuinely erased.
- *Dense* sub-bin edit: aggregate sum looks preserved, but direction agreement is near chance and
  noise ratio > 1 — not preserved per weight.

Since C-ΔΘ is sparse by construction (<5% of parameters) and RESTA is dense, this yields a
mechanistic prediction distinguishing them. Added as **H0** before any run.

### Status

- `pytest tests` → **25 passed** on CPU.
- Pre-registration frozen and committed before any experiment.
- Blocked on credentials: Hugging Face token and licence acceptances, a fresh Adaption key.
