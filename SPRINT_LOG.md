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

### Day-1 result: our GGUF pipeline reproduces Cohere's published files bit for bit

Access granted 20:28 IST; `modal run modal_app.py::day1` finished in about 5 minutes of compute
(CPU only). Full output: `results/day1_gguf_verification.json`.

| Check | Q8_0 | Q4_K_M |
|---|---|---|
| Tensors, ours vs official | 290 vs 290 | 290 vs 290 |
| Same type and shape | 290 | 290 |
| **Bit-identical raw bytes** | **290 / 290** | **290 / 290** |
| Type mix | F32 37, Q8_0 253 | F32 37, Q6_K 37, Q4_K 216 |
| Chat template (sha256 prefix) | `2147c474f41b522b` both | `2147c474f41b522b` both |
| Importance matrix used | no, both | no, both |

Built with llama.cpp `90e3841`, `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize` with
defaults and no imatrix.

**Consequences:**

1. Every survival number in this project is measured on exactly the artifact Cohere ships, not
   on an approximation of it. There is no simulation-fidelity caveat to write.
2. **Cohere ships its quantizations without an importance matrix.** For H5 the deployment-relevant
   baseline is therefore *no imatrix*. The generic imatrix and the safety-calibrated imatrix
   are both compared against it.
3. The template that injects Fire's safety preamble survives conversion unchanged. Every cell
   is evaluated under the same prompt format as the official file.

Architecture facts read from the checkpoint: 36 layers, hidden size 2048, vocabulary 262,144,
`Cohere2ForCausalLM`. `tie_word_embeddings` is absent from `config.json`; the GGUF has no separate
`output.weight` (290 = 36 × 8 + token embedding + output norm), so the output head is the tied
embedding. A repair that edits the embedding therefore edits the output head as well.

### Toolkit facts that change the repair design (read from source)

- **C-ΔΘ and RESTA add the same safety vector**, `Fire − base`. C-ΔΘ restricts it to circuit
  tensors (`layer_subset` × `target_modules`); RESTA applies it to every tensor, optionally with
  DARE (drop 30% of entries at random, rescale the rest by 1/0.7). So the H0 contrast,
  sparse vs dense, is exactly the restriction mask, with the vector held fixed. That is a cleaner
  comparison than planned.
- The SafeTune RESTA runner warns that `alpha = 1` breaks Tiny Aya; ~0.25 is suggested. Strength
  is swept on dev for both methods, as pre-registered.
- **`apply_qresafe` cannot be a baseline on this deployment path.** Its selective mode
  fake-quantizes with a symmetric per-tensor integer round-trip and returns a bf16 model, and its
  CLI path falls back to a random-probe skeleton when no calibration inputs are passed. Neither
  produces a GGUF file. Pre-registered fix 4 ("if it runs") is dropped for this reason.
- CircuitKIT registers `CohereLabs/tiny-aya-fire` in its TransformerLens port. Its shipped refusal
  task pairs each prompt with itself (`answer_contrastive`), which EAP rejects. Discovery needs a
  harmful-vs-harmless task with chat-templated prompts and single-token answers checked against
  Cohere's tokenizer. CircuitKIT's `interop_fields` writes `layer_subset` and `target_modules` in
  exactly the format `apply_ctheta` reads.

---

## 10-11 Nov 2026 — Day 2 setup: decisions made before any result

No model output has been generated or read when these were written.

### Judge: Qwen3Guard-Gen-8B primary, hosted panel secondary

The pre-registration fixes the judge *gate* (Cohen's kappa ≥ 0.7 against ~150 human labels), not
the judge. Day 0 had picked a hosted panel (`qwen3.5-397b` + `gemma-4-31b-it`). The dev screen
alone needs about 30,000 judgments (56 cells × ~550 judged responses). Through a sequential
hosted API that is many hours and an unknown share of a fixed credit budget.

**Primary judge is now `Qwen/Qwen3Guard-Gen-8B`**, at revision `4505cb1a`, converted to Q8_0 by the
pinned llama.cpp and served on the same GPU stack as the models under test. Reasons:

- It is open-weight under Apache 2.0, so anyone can reproduce the evaluation without a paid API.
- It is multilingual (119 languages, Hindi included).
- One call returns both labels the study needs: `Safety` (harmful compliance) and `Refusal`
  (over-refusal on benign prompts).

The hosted panel is kept as a second opinion on a subset. The kappa gate applies to whichever
judge is primary.

### English red-team prompts: semantic translation where given, literal otherwise

The first data build failed loudly: only 91 of 915 Hindi `aya_redteaming` rows have a
`semantic_translation`. All 915 have a `literal_translation`. The annotators add a semantic
translation only where the literal one misses the meaning. The English condition therefore uses
the semantic translation when present and the literal one otherwise. Each English row records
which field it came from (`en_from`), so H2 can be checked on the 91 semantically translated items
alone if the literal ones look unnatural.

### CUDA build: static linking

The GPU image failed at link time: Modal's image builder has no GPU driver, so a shared
`libggml-cuda.so` cannot resolve `libcuda.so.1` when the executables are linked. Building with
`-DBUILD_SHARED_LIBS=OFF` links against the toolkit's driver stub instead, and the binaries pick
up the real driver at run time. The commit is unchanged (`90e3841`).

### Drift: both pre-registered variants trained at once

The pre-registered gate is: plain Hindi fine-tuning must raise Hindi harmful compliance by at
least 10 points, otherwise a fixed 2–5% harmful dose is added. Both variants are trained in
parallel so the gate costs no extra wall-clock time:

- `plain`: 0% dose.
- `dose3`: 3% dose, the middle of the range.

The gate rule, unchanged, decides which one is used. The harmful dose is BeaverTails (SafeTune's
default harmful corpus), decontaminated against every evaluation prompt like the Hindi set. Its
text stays on the private volume; the repository records only row indices.

Training settings: SafeTune `PlainSFTTrainer`, LoRA r=16, alpha=32 on all seven projections,
3 epochs, lr 2e-4, batch 4, max length 1024, seed 0. Tokenizer, chat template and configs are
restored from Fire after the merge, and the merged tensor layout is checked against Fire's.

### Repair selection rules, fixed before any repaired model exists

**Circuit size for C-ΔΘ.** SafeTune's `apply_ctheta` edits the cross product of CircuitKIT's
`layer_subset` and `target_modules`, so the share of parameters edited depends on how many top
nodes are kept. C-ΔΘ is described as editing under 5% of parameters. **Rule:** use the largest
`top_fraction` in {0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01} whose edit,
counted with SafeTune's own key-matching rule, is at most 5% of parameters.

**Answer tokens for circuit discovery** come from Fire, not from CircuitKIT's defaults (" I" /
" Sure", chosen for GPT-2 and Llama). Per language, the refusal token is the greedy first token
whose share rises most from harmless to harmful calib prompts, and the compliance token is the
reverse. TransformerLens-vs-Hugging Face parity (next-token KL) is recorded before the circuit
is used.

**Strength sweep** on dev, BF16 only. RESTA alpha in {0.1, 0.25, 0.5}; C-ΔΘ strength in
{0.5, 1, 2}. C-ΔΘ edits fewer tensors, so it needs a larger multiplier to move behaviour. For
each method, choose the **smallest strength that restores at least 50% of the drift** in Hindi
harmful compliance, provided it does not raise XSTest over-refusal by more than 10 points or
lower MMLU by more than 5 points versus the drifted model. If no strength qualifies, take the
one with the largest restoration and report that the pre-registered gate failed. Then fall back
to LoX or SafeMerge on dev, as pre-registered.

### Drift training finished (no evaluation yet)

| Variant | Train rows | Harmful dose | Loss, first → last | Runtime | Layout = Fire |
|---|---|---|---|---|---|
| `plain` | 600 | 0 | 1.661 → 0.874 | 302 s | yes, 290 tensors |
| `dose3` | 600 | 18 (2.9%) | 1.653 → 0.768 | 319 s | yes, 290 tensors |

The curated Hindi set has 600 training pairs. The source has 1,153 Hindi rows; 128 failed the
Devanagari check, 22 were exact and 99 near duplicates, 3 were over length and 1 overlapped an
evaluation prompt. 200 are held out for perplexity and 100 kept as harmless calibration prompts.

---

## 11 Nov 2026 — Day 2: Fire baseline dev cells, and a decoding bug found

First real evaluation: Fire at BF16 and Q4_K_M on the dev split, judged by Qwen3Guard-Gen-8B.
The pipeline works end to end (judge parsed 100% of responses, perplexity ran, the survival tool
is unused here). But the generations exposed a decoding problem that must be fixed before any
result is trusted.

### What is fine

- **Hindi generation is coherent.** Red-team Hindi responses are fluent, correctly refuse or
  answer, and stop cleanly on `<|END_OF_TURN_TOKEN|>`. Degenerate rate 0.01-0.02. The chat
  template injects Fire's "contextual safety mode" preamble exactly as intended
  (`rendered_prompt_example.txt` confirms it).
- **The judge works.** `judge_parsed_rate` 1.0 on every task; it returns Safety and Refusal
  labels as designed.
- **The degenerate-output detector works.** It flagged the repetition loops below rather than
  letting them be scored as refusals.

### The bug: greedy decoding loops, worst on short answers

I served every model with `temperature=0` (greedy) for reproducibility and set llama.cpp's
default repetition penalty, which is **off** (1.0). On this model family that produces loops:

- **MMLU is broken.** English accuracy 0.29-0.30 with **49-55% of answers unparseable**; outputs
  like `"Cff >f >f >f"`, `"D in in in in"`, `"Bee ife ife"`. The model emits a letter then
  loops filler, and the 24-token cap catches it mid-loop. Hindi MMLU is less bad (0.42, 20-28%
  unparseable) but still hurt.
- **English red-team / XSTest** show degenerate rates 0.15-0.20 (vs 0.01-0.02 in Hindi) and
  near-total truncation, because looping runs to the token cap.

This is the known greedy-plus-no-penalty failure mode, not a template or quantization problem
(Hindi on the same files is clean). It bites hardest where the intended answer is short, which
is exactly MMLU.

### Fix (applied to the code, not yet re-run)

`serve.LlamaServer.chat` now sends `repeat_penalty` (default 1.1) and `repeat_last_n`, keeping
greedy decoding so cells stay reproducible. The judge path (`complete`) is left unchanged; it
already parsed cleanly and a penalty could distort its structured labels. Next session, before
spending on the grid:

1. Read Fire's `generation_config.json` from the volume and prefer its sampling settings if it
   defines them.
2. Re-run only the two Fire dev cells and confirm the MMLU unparseable rate collapses and the
   English degenerate rate drops to the Hindi level.
3. Only then build the drift/repair ladders' cells.

Nothing downstream is invalidated because no result has been accepted yet; this is the day-2
dev screen catching a setup bug before any budget went to the full grid, which is what the dev
screen is for.

### Numbers recorded (provisional, decoding bug present — not to be quoted)

| Cell | redteam-hi unsafe | redteam-en unsafe | xstest-en refusal | MMLU-hi acc | PPL (hi) |
|---|---|---|---|---|---|
| Fire BF16 | 0.027 | 0.040 | 0.15 | 0.42 | 8.40 |
| Fire Q4_K_M | 0.040 | 0.020 | 0.17 | 0.36 | 8.61 |

Perplexity rises slightly BF16 -> Q4_K_M (8.40 -> 8.61), as expected. These values are logged for
pipeline provenance only and are superseded once decoding is fixed.
