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

### Toolchain verified on Modal (CPU, ~3 minutes, a few cents)

`modal run modal_app.py::smoke_all` built all four images and probed them. Four separate images
because the toolkits do not share a dependency set; one image would mean resolving conflicts that
do not need resolving.

| Image | Python | Result |
|---|---|---|
| safetune | 3.12.10 | torch 2.9.0+cu128, transformers 5.22.2, **safetune 0.1.14**, `apply_ctheta` importable |
| circuitkit | 3.12.10 | **circuitkit 0.2.2**, `discover` and `quantize` both present |
| eval | 3.12.10 | **auditkit 1.1.7**, **curatorkit 0.1.24** co-install cleanly |
| llamacpp | 3.12.10 | `llama-quantize`, `llama-cli`, `llama-imatrix`, `convert_hf_to_gguf.py` all built |

All six pre-registered quant types are present in the built binary: Q8_0, Q6_K, Q4_K_M, Q4_0,
Q3_K_M, Q2_K.

**Both fixes are feasible, confirmed against the binary rather than the documentation:**
`--tensor-type` exists (fix 1, circuit-guided precision) and `--imatrix` exists (fix 2,
safety-calibrated importance matrix). Had either been missing, two of the four planned fixes would
have had to be redesigned.

One probe failed harmlessly: `circuitkit.models.architecture_registry` was a guess at a module
path and does not exist. Whether cohere2 is in CircuitKIT's registry is still unverified and is a
day-1 check.

### Status at end of day 0

- `pytest tests` → **25 passed** on CPU.
- Pre-registration frozen and committed (`5ba4cba`) before any experiment existed.
- Toolchain proven end to end except for model loading.
- Modal authenticated as profile `shashankchoudhary792`.
- **Blocked on credentials:** Hugging Face token plus licence acceptances for the gated Tiny Aya
  and aya_redteaming repos.

### Adaption judge endpoint verified

`.env` was initially saved by Windows as `.env.txt` (Notepad appends the extension silently, and
Explorer hides it). Two consequences, both fixed:

1. Code looking for `.env` did not find it.
2. **`.env.txt` was not matched by the `.gitignore` rule `.env`**, so the key was briefly at risk
   of being committed. The ignore rule is now widened to `.env`, `.env.*`, `*.env`, `*.key`,
   `*.pem`, `secrets.*`. Verified with `git check-ignore` and `git ls-files`.

`ADAPTION_BASE_URL` was absent (the dashboard issues a key only; the base URL lives in the
provider's docs). Recovered from the previously working swarm-threshold integration, which proved
the surface: `POST {base}/responses` and `GET {base}/models?type=hosted`, both with
`Authorization: Bearer <key>`.

Verified with a model listing, the cheapest possible call (no generation): **HTTP 200**, 7 hosted
models reachable — `deepseek-v4-flash`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `glm-5.2`,
`glm-53-flash`, `qwen3.5-397b`, `qwen3.8-27b`.

**Judge panel chosen:** `qwen3.5-397b` and `gemma-4-31b-it` as two independent judges, `glm-5.2`
as tie-breaker. None is trusted until it clears the κ ≥ 0.7 gate against ~150 human labels.

Note: the supplied key matches the length and prefix of the one that appeared in a tool output
earlier in this session, so it is probably the same key rather than a rotated one. It works;
rotation is still advisable.

---

## 10 Nov 2026 — Day 1, gated-model access

### A false positive in our own access check

The first day-1 run failed at the first download:
`GatedRepoError: 403 … Access to model CohereLabs/tiny-aya-fire is restricted and you are not in
the authorized list.` The pre-flight check had reported the repo as accessible. It used
`HfApi.model_info()`, and the Hub returns metadata, file list included, for gated repos whether
or not the account has access. That check could never fail, so it proved nothing.

Replaced with `get_hf_file_metadata(hf_hub_url(repo, file))`, which requests the headers of a real
file and fails exactly when a download would. Result:

| Repo | Gating | Download access |
|---|---|---|
| `CohereLabs/tiny-aya-fire` | `auto` | **blocked**: licence not yet accepted by this account |
| `CohereLabs/tiny-aya-base` | `auto` | **blocked**: same |
| `CohereLabs/aya_redteaming` | not gated | OK |
| `CohereLabs/tiny-aya-fire-GGUF` | not gated | OK |

The token is not the problem: it is fine-grained with `canReadGatedRepos: True`. Both models use
automatic approval, so access starts the moment the account owner submits the licence form.

### Day-1 job hardened before its first real run

- Pre-flights download access and stops with the fix in plain words instead of a traceback.
- Verifies **two** quant types against Cohere's published files instead of one. Q8_0 takes no
  importance matrix in llama.cpp, so a Q8_0 mismatch isolates the converter. Q4_K_M is the
  default public download.
- Compares raw tensor bytes (exact), then sizes any differences on the largest differing tensors.
- Records the GGUF metadata that explains a mismatch: `quantize.imatrix.*` (did Cohere use an
  importance matrix?), file type, quantization version, and a hash of the chat template.
- Downloads `*.jinja` as well. Transformers 5 stores the chat template in `chat_template.jinja`;
  without it our GGUF would ship without the template that injects Fire's safety preamble, and
  every behavioural number would be measured on a different prompt format from the official file.
- Writes outputs through `.partial` files and renames on success, so an interrupted run cannot
  leave a truncated GGUF that a later run mistakes for finished.
