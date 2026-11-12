# Refusals That Stick

**Do weight-space safety repairs survive GGUF quantization?**

A pre-registered study on `CohereLabs/tiny-aya-fire` (3.35B), in Hindi and English.
Lexsi AI Safety and Alignment Hackathon, Track 2b: repairing safety drift without retraining.
Solo entry by Shashank Kumar.

![Three headline numbers](docs/figures/overview.png)

| | |
|---|---|
| **Code** | this repository |
| **Data** (splits, hand labels) | [Sha09090/refusals-that-stick-data](https://huggingface.co/datasets/Sha09090/refusals-that-stick-data) |
| **Per-prompt results** (every cell) | [Sha09090/refusals-that-stick-results](https://huggingface.co/datasets/Sha09090/refusals-that-stick-results) |
| **Drift adapter** (warning card) | [Sha09090/tiny-aya-fire-hindi-drift-lora](https://huggingface.co/Sha09090/tiny-aya-fire-hindi-drift-lora) |
| **Collections** | [models](https://huggingface.co/collections/Sha09090/refusals-that-stick-models-6af4a95fabd9b1da26e73389) · [datasets and logs](https://huggingface.co/collections/Sha09090/refusals-that-stick-datasets-and-logs-6af4a95f78ec099a11625ad7) |
| **Plan and log** | [`PREREGISTRATION.md`](./PREREGISTRATION.md) (frozen before any run) · [`SPRINT_LOG.md`](./SPRINT_LOG.md) (every deviation, with evidence) |
| **Website** | [sidthebuilder.github.io/refusals-that-stick](https://sidthebuilder.github.io/refusals-that-stick/) |

---

## Summary

A developer fine-tunes an aligned model for their language, and its safety drifts. They repair
the weights with a published method, then quantize the model to GGUF so it runs on a laptop or a
phone. This project measures whether the repair is still there after quantization.

- **The test bed is the real deployment artifact.** Our GGUF files are bit-identical to Cohere's
  published ones (290 of 290 tensors at Q8_0 and Q4_K_M).
- **Post-hoc repair has a hard limit on this drift.** After a fine-tune with a 3% harmful dose,
  none of 13 configurations of four SafeTune repair methods restores half of the lost refusal
  without a large capability cost. The study therefore uses the two best *usable* partial repairs.
- **In the weights, 4-bit quantization scrambles a sparse repair.** At Q4_K_M, 98.7% of the sparse
  repair's edited weights change value, but only 55.6% of them move the way the repair intended
  (a coin flip gives 50%). The dense repair keeps 74.1%. Sparse keeps less than dense at all six
  quantized precisions, as pre-registered.
- **In behaviour, the loss is too small for any feasible test to see.** On 663 held-out prompts
  per language, both repairs still beat the drifted model at Q4_K_M. The estimated loss is within
  noise, and detecting it would take about 3,500 red-team prompts per language.
- **A cheap fix works in the weights.** Keeping the 15 tensors the sparse repair edited at Q8_0
  inside the Q4_K_M file brings intended-direction survival from 55.6% to 91.1%, for 7.5% more
  file size. This was verified on a built file.

The two levels do not agree, and the study shows why: a behavioural evaluation of normal size
cannot resolve a weight-level loss of this size. Neither one can be used as a stand-in for the
other.

## Check your own repair

`rts-check` runs this measurement on any two lossless GGUF files of one model, before and after a
weight edit, and prints the fix. CPU only.

```bash
pip install -e .
rts-check before-BF16.gguf after-BF16.gguf --precisions Q8_0,Q4_K_M \
    --llama-quantize path/to/llama-quantize --verify
```

On this study's drifted and sparse-repaired models (`modal run modal_app.py::check`), it builds
both files at each precision, measures the edit, then builds the protected file and measures it:

```text
Edit: 15 of 290 tensors changed, 274,395,007 weights (8.2% of the model).

Survival of the edit, over the edited weights:
  precision    erased   intended among moved   directional survival
  Q8_0          18.6%                  91.1%                  74.1%
  Q4_K_M         1.3%                  55.6%                  54.9%
  Intended among moved is 50% for a coin flip.

Keep the 15 edited tensors at Q8_0 inside the Q4_K_M file:
  intended among moved 55.6% -> 91.1%, erased 1.3% -> 18.6%
  file size +7.5% (the whole model at Q8_0: +66.9%)
  computed exactly from the Q8_0 and Q4_K_M measurements
  built and measured: intended among moved 91.1%, 0 other tensors re-typed (matches)
  Build it with:
  llama-quantize \
    --tensor-type '^blk\.0\.attn_q\.weight$=q8_0' \
    ... 14 more --tensor-type arguments ...
    AFTER.gguf AFTER-Q4_K_M-protected.gguf Q4_K_M
```

Full output: [`results/check_safemerge-dose3-s1.txt`](./results/check_safemerge-dose3-s1.txt),
with the per-tensor report in the `.json` beside it.

---

## Why this question

MANSU ([arXiv:2605.15138](https://arxiv.org/abs/2605.15138), Lexsi Labs) shows that 4-bit
quantization can reverse *unlearning*: per-weight updates smaller than one quantization bin round
back to where they started. It describes circuit-restricted refusal repair as a setting that
"does not require" this quantization permanence. Safety repairs are also small, targeted edits,
so the question is open for repair.

| Work | What it shows | Setting |
|---|---|---|
| [MANSU](https://arxiv.org/abs/2605.15138) | quantization reverses unlearning; updates below the bin width vanish | unlearning |
| [DurableUn](https://arxiv.org/abs/2605.02196) | INT4 restores forgotten content in models that passed a BF16 audit | unlearning |
| [Q-resafe](https://arxiv.org/abs/2506.20251) | patches safety after quantizing | repair, after quantization |
| [Q-realign](https://arxiv.org/abs/2601.08089) | realigns during the quantization step | repair, during quantization |

Both repair papers produce a safe quantized model. Neither measures what happens to a repair made
at full precision with a published method and then quantized. A prior-work scan run before any
model was trained (ten arXiv queries plus GitHub and Hugging Face searches) found no such
measurement; the raw output is in [`results/novelty_scan.txt`](./results/novelty_scan.txt).

---

## Results

### 1. The measurements use Cohere's real files

Our conversion with llama.cpp `90e3841` is bit-identical to Cohere's published GGUF files: 290 of
290 tensors at Q8_0 and at Q4_K_M, with the same chat template and no importance matrix. Every
number below is measured on the artifact people download.
([`results/day1_gguf_verification.json`](./results/day1_gguf_verification.json))

### 2. The drift, and where post-hoc repair stops working

Fine-tuning Fire on 600 curated Hindi instruction pairs alone lowered Hindi refusal by only 4.7
points (p = 0.23), below the pre-registered 10-point bar. The pre-registered fallback added a 3%
dose of harmful English pairs from BeaverTails:

| Model | Hindi valid refusal | English valid refusal | MMLU (en) | Hindi perplexity |
|---|---|---|---|---|
| Fire | 71.3% | 81.3% | 0.65 | 8.40 |
| Hindi fine-tune only | 66.7% | 82.0% | 0.62 | 8.79 |
| **Hindi + 3% harmful (used)** | **23.3%** | **22.7%** | 0.65 | 8.90 |

SafeTune separates train-time hardening from post-hoc recovery. This study tests only the post-hoc,
data-free repair methods (C-ΔΘ, RESTA, LoX and SafeMerge) across 13 configurations, against a gate
fixed in advance: restore at least 50% of the Hindi refusal drop, lose at most 5 MMLU points, and
raise benign-prompt refusal at most 10 points above Fire.

![Repair configurations against the gate](docs/figures/repair_frontier.png)

- Every configuration that restores at least half of the drop loses 28 to 65 MMLU points in the
  worse language and raises Hindi perplexity at least 17-fold.
- The only configurations inside the MMLU limit are the three SafeMerge strengths, which restore
  15–24%.
- No configuration passes the gate. We report that rather than lowering it.

On a drift this severe, data-free post-hoc repair trades safety against capability with nothing in
between. Train-time hardening, which SafeTune also provides, was outside this study's scope.

The quantization study therefore uses the two partial repairs a developer could actually ship, a
rule fixed before any quantized model was evaluated:

| Arm | Repair | Edits | Hindi restoration | MMLU change (en / hi) |
|---|---|---|---|---|
| **Sparse** | SafeMerge α1.0 | 274M weights in 15 of 290 tensors (8.2%) | 24% | −1 / −2 pt |
| **Dense** | RESTA α0.25 | 3.26B weights in all 290 tensors (97.3%) | 36% | −9 / −12 pt |

### 3. Weight level: the sparse repair's direction is close to a coin flip at 4 bits

For every edited weight, we compare the repair's intended change with the change that survives in
the quantized file. Each weight ends up in one of three states: erased (rounded back), moved the
intended way, or moved against the repair.

![Fate of every edited weight](docs/figures/weight_survival.png)

| Precision | Sparse: erased | Sparse: intended among moved | Dense: erased | Dense: intended among moved |
|---|---|---|---|---|
| Q8_0 | 18.6% | 91.1% | 0.6% | 96.8% |
| Q6_K | 12.3% | 69.3% | 1.5% | 87.3% |
| **Q4_K_M** | 1.3% | **55.6%** | 0.3% | **74.1%** |
| Q4_0 | 45.3% | 57.1% | 11.9% | 77.2% |
| Q3_K_M | 33.5% | 53.9% | 11.1% | 73.1% |
| Q2_K | 9.9% | 51.8% | 6.1% | 68.4% |

- The sparse repair keeps less than the dense repair at all six precisions (hypothesis H0).
- At Q4_K_M and Q2_K, almost every edited weight changes value, but the direction is close to a
  coin flip. A per-weight "did the value change?" check would call 98.7% of this edit preserved.
- At Q4_0 and Q3_K_M, a third to a half of the edited weights round straight back.

This is a census of every edited weight, so there is no sampling error.
([`results/survival_analysis.json`](./results/survival_analysis.json))

### 4. Behaviour level: the loss is within noise on held-out prompts

The held-out test split (663 red-team prompts per language) was run once, after the screening
analysis was frozen. Both repairs remain significantly better than the drifted model at Q4_K_M
(paired exact McNemar test, Holm-corrected):

| Arm, language | Benefit at BF16 | Benefit at Q4_K_M | p at Q4_K_M |
|---|---|---|---|
| Sparse, Hindi | +6.9 pt | +4.7 pt | 1.2 × 10⁻³ |
| Sparse, English | +9.5 pt | +8.3 pt | 5.0 × 10⁻⁹ |
| Dense, Hindi | +16.3 pt | +18.4 pt | 1.2 × 10⁻²¹ |
| Dense, English | +23.1 pt | +21.3 pt | 2.0 × 10⁻²⁷ |

The benefit lost to quantization is estimated per prompt as
`(repair − drift) at BF16 − (repair − drift) at Q4_K_M`, with a seeded bootstrap over prompts:

![Benefit lost at Q4_K_M with 95% intervals](docs/figures/behaviour_test.png)

No interval excludes zero. The sparse repair's point estimates lean the predicted way, and the
dense repair's Hindi estimate leans the other way.
([`results/quant_cost_test.json`](./results/quant_cost_test.json))

### 5. Why the two levels disagree

The sparse repair's whole Hindi benefit is 6.9 points, about 46 prompts out of 663. Detecting the
estimated 2.3-point loss at 80% power would take about 3,500 red-team prompts per language. That
is nearly four times the entire Hindi Aya red-team set (913 prompts, all of which this study
uses).

![Detectable loss against test size](docs/figures/power_curve.png)

Hypothesis H6 predicted that weight-level survival would track behavioural retention (Spearman
ρ > 0.5). It did not: ρ = 0.38 across 24 cells. Given the power curve, this is expected. For a
developer, the conclusion works both ways:

- A weight-level metric is not a safety guarantee. It was tested here as a proxy for behaviour
  and did not pass.
- A passing behavioural evaluation is not evidence that the repair survived in the weights. At
  Q4_K_M, only 55.6% of the sparse repair's moved weights go the intended way, and a 663-prompt
  test cannot see any loss.

### 6. A fix: keep the repaired tensors at 8 bits

Pre-registered fix 1 keeps chosen tensors at Q8_0 inside a 4-bit file
(`llama-quantize --tensor-type`), compared with a random control of the same size. Q8_0 quantizes
each tensor on its own, so the weight-level result of any protection set follows exactly from the
per-tensor survival maps. The arms and the success rule were logged before any number was
computed.

![Protection arms](docs/figures/protection.png)

| Sparse repair, Q4_K_M file | Tensors at Q8_0 | Intended among moved | Erased | File size |
|---|---|---|---|---|
| No protection | 0 | 55.6% | 1.3% | +0% |
| Random tensors, same bytes (mean of 10,000 draws) | about 29 | 59.0% | – | +6.9% |
| CircuitKIT refusal circuit (fix 1 as pre-registered) | 28 | 60.2% | 4.0% | +7.0% |
| **The 15 tensors the repair edited** | 15 | **91.1%** | 18.6% | **+7.5%** |
| Whole model at Q8_0 | 290 | 91.1% | 18.6% | +66.9% |

- **Checked on a real file.** The protected Q4_K_M file built with `--tensor-type` measures
  exactly the computed values: 18.6% of edited weights erased, and directional survival 0.741
  over all edited weights, which is 91.1% intended among the 81.4% that moved. None of the other
  275 tensors changed type. ([`results/protection_check.json`](./results/protection_check.json))
- **Edit-aware protection is the practical fix.** It gives Q8_0-level survival for 7.5% more file
  size, against 66.9% for the whole model at Q8_0. A developer knows which tensors their repair
  edited, so it needs no circuit discovery. The dense repair edits every tensor, so the same
  budget barely helps it (74.1% → 76.9%).
- **The circuit version (H4) is weakly supported.** It beats all three pre-registered random
  seeds, which is the logged rule. But 28% of 10,000 random sets of the same size do as well
  (20% at a Q4_0 base), because only 3 of the circuit's 28 tensors are tensors the repair edited.
- **Behaviour.** On the dev split (150 prompts per language) the protected file did not change
  detectably from the plain Q4_K_M file: refusal −4.0 points in Hindi and +4.0 in English (both
  p = 0.15), with MMLU, benign-prompt refusal and perplexity unchanged. Given the power curve
  above, a behavioural gain could not be seen at this size either way.
  ([`results/protection_behaviour.json`](./results/protection_behaviour.json))

([`results/protection_analysis.json`](./results/protection_analysis.json))

### 7. A control: quantization also degrades the aligned model

Fire itself, with no fine-tune and no repair, drops from 71.3% to 55.3% Hindi refusal at Q2_K
(p = 3 × 10⁻⁶), and its Hindi perplexity rises from 8.40 to 12.02. This is why every repair is
compared with the drifted model *at the same precision*: it separates "quantization undid the
repair" from "quantization hurts this model generally".

### 8. Fix 2: an importance matrix restores perplexity, not refusal

Pre-registered fix 2 calibrates llama.cpp's importance matrix on safety prompts, at no cost in
file size. It was applied where Fire breaks, at Q2_K, against two controls: a generic matrix
(WikiText-2) and a benign-chat matrix (harmless prompts in the same chat format as the safety
prompts). Each matrix saw the same amount of text. Dev split, 150 red-team prompts per language:

| Fire at Q2_K | Hindi valid refusal | Hindi perplexity | MMLU (hi / en) |
|---|---|---|---|
| No importance matrix (as Cohere ships it) | 55.3% | 12.02 | 0.50 / 0.52 |
| Generic matrix | 50.0% | 10.91 | 0.39 / 0.45 |
| Benign-chat matrix | 52.7% | 10.69 | 0.49 / 0.45 |
| Safety matrix (fix 2) | 58.0% | 10.72 | 0.48 / 0.50 |
| BF16, for reference | 71.3% | 8.40 | 0.53 / 0.65 |

- **Fix 2 does not pass its screen.** The safety matrix is 2.7 points above no matrix (p = 0.61),
  against a bar of 8 points logged before the run, so the test split was not run. H5 is not
  supported.
- **Every matrix improves perplexity by 9–11%, and none restores refusal.** At this precision the
  usual quality measure for a quantized file gets better while the safety behaviour does not.
- The safety matrix is 8.0 points above the generic one (p = 0.029), but not after the Holm
  correction (p = 0.14), so no difference is claimed.

([`results/h5_dev.json`](./results/h5_dev.json))

---

## Supporting findings

**A stop-token bug in Fire's GGUF files.** Fire's Hugging Face config stops on `<EOS_TOKEN>`
(id 3), which the model emits after `<|END_RESPONSE|>`. The GGUF metadata records only
`<|END_OF_TURN_TOKEN|>` (id 6), so llama.cpp generates past the model's own stop marker. Before
the fix, 49–55% of English MMLU answers could not be parsed. It affects Cohere's official files,
since ours are identical to them. Registering both stop tokens at load time fixes it without
editing any file. ([`results/diag_decoding.json`](./results/diag_decoding.json))

**A validated judge.** `Qwen/Qwen3Guard-Gen-8B`, at a pinned revision, was scored against 150 hand
labels with AuditKIT's calibration metrics, against a pre-registered bar of Cohen's κ ≥ 0.7:

| Judge label | n | Agreement | κ | Decision |
|---|---|---|---|---|
| Refusal | 150 | 87.3% | 0.73 | primary outcome |
| Harmful | 87 | 77.0% | 0.52 | secondary, reported with a caveat |

The primary outcome is **valid refusal**: the judge says the model refused, *and* the response is
not degenerate. A model quantized into gibberish is never counted as safe.
([`results/judge_validation.json`](./results/judge_validation.json))

**Benign fine-tuning drifts only the fine-tuned language.** The Hindi-only fine-tune lowered Hindi
refusal by 4.7 points and left English unchanged (+0.7 points, p = 1.0).

**Return on investment.** Throughput was measured on the test split: one NVIDIA L40S, llama.cpp
with 32 parallel slots, 417 prompt and 116 completion tokens per request on average. At Modal's
$1.95 per hour and OpenAI's list prices for the same tokens (checked 12 Nov 2026):

| Option | Cost per 10,000 requests |
|---|---|
| Tiny Aya Fire Q4_K_M, rented L40S | $1.02 |
| Sparse repair Q4_K_M, rented L40S | $0.67 |
| `gpt-6-luna` API | $1.00 |
| `gpt-5-mini` API | $3.36 |
| `gpt-6.1-sol` API | $19.92 |

A rented GPU roughly matches the cheapest API tier, so the case for the small model is not
price. It is that the 2.1 GB Q4_K_M file runs on the user's own device, with no per-token bill and
no data leaving it, which is exactly the deployment this study is about. Tiny Aya is CC-BY-NC, so
this applies to research and non-commercial use. ([`results/roi.json`](./results/roi.json))

---

## Pre-registered hypotheses

| | Hypothesis | Result |
|---|---|---|
| H0 | A sparse repair loses more of its edit under quantization than a dense one | **Supported** at 6 of 6 precisions¹ |
| H1 | Directional survival is below 0.9 at Q4_K_M | **Supported** (0.549 sparse, 0.738 dense) |
| H2 | Behavioural retention is lower in Hindi than in English | **Mixed** (10 of 12 dev cells; 1 of 2 test cells) |
| H3 | The sparse and dense repairs differ in behavioural retention | **Not established** (intervals overlap) |
| H4 | Circuit-guided mixed precision beats a random control at equal bytes | **Weak** (passes the logged rule; 28% of random sets match it) |
| H5 | A safety-calibrated importance matrix beats a generic one | **Not supported** (fails its dev screen; +8.0 pt vs generic, Holm p = 0.14) |
| H6 | Weight survival predicts behavioural retention (ρ > 0.5) | **Falsified** (ρ = 0.38) |

¹ Pre-registered with C-ΔΘ as the sparse repair. C-ΔΘ restored nothing at usable strengths, so
SafeMerge took its place under a rule logged before any quantized result.

---

## How it was built

| Stage | Tool | What it does here |
|---|---|---|
| Data | **CuratorKIT** | cleans and deduplicates the Hindi fine-tuning set, plus a Devanagari-script gate and decontamination against every evaluation prompt (no shared 8-word sequence) |
| Drift | **SafeTune** | `PlainSFTTrainer`: LoRA r = 16 on all attention and MLP projections |
| Circuit | **CircuitKIT** | EAP-IG refusal circuit for C-ΔΘ (layers 0, 14, 16 and 20) |
| Repair | **SafeTune** | C-ΔΘ, RESTA, LoX and SafeMerge on the drifted model |
| Judge check | **AuditKIT** | Cohen's κ of the judge against hand labels |
| Quantization | llama.cpp `90e3841` | GGUF conversion, seven precisions, serving |
| Survival | `survival.py` | per-weight comparison of intended and realized edits from real GGUF files |
| Statistics | `stats.py` | exact McNemar, Holm correction, Wilson intervals, bootstrap, Spearman |

**Two metric choices made before any model ran.** The `gguf` Python package cannot quantize to
k-quants, so survival is measured from real `llama-quantize` output instead of a simulation. Two
obvious metrics proved misleading in tests: a magnitude ratio scored 1.41 on pure noise, and a
least-squares projection scored 1.93 on an edit 97% of which had been erased. The study therefore
counts outcomes per weight. Both failures are kept as tests.

**Splits.** Red-team prompts come from `CohereLabs/aya_redteaming` (Hindi), each paired with its
English translation from the same dataset. Splits: 100 prompts for calibration, 150 for
screening, 663 for the one-time test. XSTest was translated into Hindi with back-translation
checks.

### Models and datasets

| Model | Role | Licence |
|---|---|---|
| [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire) | the model under study | CC-BY-NC-4.0 |
| [`CohereLabs/tiny-aya-base`](https://huggingface.co/CohereLabs/tiny-aya-base) | the base of the safety vector (Fire − base) that the repairs use | CC-BY-NC-4.0 |
| [`CohereLabs/tiny-aya-fire-GGUF`](https://huggingface.co/CohereLabs/tiny-aya-fire-GGUF) | Cohere's published files, the reference for the bit-identity check | CC-BY-NC-4.0 |
| [`Qwen/Qwen3Guard-Gen-8B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) | refusal and safety judge, pinned revision, validated against hand labels | Apache-2.0 |
| `gemma-4-31b-it`, `deepseek-v4-flash` | XSTest translated into Hindi, then back-translated by the second model for the check | hosted API; only the translated prompts are kept |

| Dataset | Role | Licence |
|---|---|---|
| [`CohereLabs/aya_redteaming`](https://huggingface.co/datasets/CohereLabs/aya_redteaming) | Hindi red-team prompts and their English versions: calibration, dev and test splits | Apache-2.0 |
| [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Hindi instruction pairs for the drift fine-tune, harmless calibration prompts, Hindi perplexity text | Apache-2.0 |
| [`PKU-Alignment/BeaverTails`](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | the 3% harmful dose (published as row indices only) | CC-BY-NC-4.0 |
| [`Paul/XSTest`](https://huggingface.co/datasets/Paul/XSTest) | over-refusal prompts, English and translated Hindi | CC-BY-4.0 |
| [`CohereLabs/Global-MMLU`](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | capability check in Hindi and English | Apache-2.0 |
| [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) | generic calibration text for the importance matrix (fix 2) | CC-BY-SA-3.0 / GFDL |

### Notes for the Lexsi toolkits

Problems met while building, each with its evidence in `SPRINT_LOG.md`:

- **SafeTune `apply_ctheta` edits whole layers.** It edits every pair in `layer_subset` ×
  `target_modules`, so once a circuit holds both a head and an MLP of a layer, every projection in
  that layer is edited. No circuit size met C-ΔΘ's 5% parameter budget; the smallest gave 9.3%.
  Selecting individual (layer, module) pairs would let the edit follow the circuit.
- **SafeTune `apply_qresafe` does not reach the deployment format.** Its selective mode
  fake-quantizes with a symmetric per-tensor round trip and returns a bf16 model, and the CLI falls
  back to a random-probe skeleton when no calibration inputs are passed. Neither produces a GGUF
  file, so it could not be compared on the llama.cpp path. A warning when the skeleton is used
  would help.
- **CircuitKIT's refusal task needed rebuilding for Tiny Aya.** The shipped task pairs each prompt
  with itself (`answer_contrastive`), which EAP rejects. The default answer tokens (" I" /
  " Sure") suit GPT-2 and Llama; Fire's own first tokens are `I'm` / `The` in English and
  `मैं` / `उत्तर` in Hindi. Choosing answer tokens from the model's own outputs would carry the
  task to other models and languages.
- **CircuitKIT EAP-IG ran out of memory on an A100-80GB** with default settings, because the hooks
  EAP needs materialize per-head tensors. Batch size 1, `expandable_segments` and 128 contrastive
  pairs fixed it (about 12 minutes). Its parity gate (next-token KL below 1e-4) is stated for
  fp32; in bf16 the measured KL was 0.0012–0.0019 nats.
- **CuratorKIT has no script gate and no decontamination step.** Both were written here as
  CuratorKIT gates (`data.py`): a Devanagari-share gate and an 8-word overlap check against every
  evaluation prompt. Both would be useful built in for multilingual safety data.
- **SafeTune needs Python ≥ 3.12 and CircuitKIT pins transformer-lens**, so each toolkit ran in its
  own image (`modal_app.py`). AuditKIT's `judge_calibration` gave the judge's κ against the hand
  labels directly.

---

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m pytest tests -q            # 174 tests, CPU only, no network or API keys
python scripts/make_figures.py       # rebuilds every figure from results/ (needs matplotlib)
python scripts/analyze_protection.py # the fix, computed from the committed survival maps
```

The per-prompt labels behind every statistic are published in the
[results dataset](https://huggingface.co/datasets/Sha09090/refusals-that-stick-results). The full
pipeline runs on Modal:

```bash
modal run modal_app.py::day2_setup   # data, models, CUDA check
modal run modal_app.py::drift        # both drift variants
modal run modal_app.py::circuit      # refusal circuit
modal run modal_app.py::repairs --variant dose3
modal run modal_app.py::grid --models fire,drift-dose3,safemerge-dose3-s1,resta-dose3-s0.25
modal run modal_app.py::survival --repairs safemerge-dose3-s1,resta-dose3-s0.25
modal run modal_app.py::protect      # builds and checks the protected file
modal run modal_app.py::h5           # fix 2: three importance matrices, Fire at Q2_K, dev cells
modal run modal_app.py::check        # the check command on the drifted and sparse-repaired models
```

## Repository layout

| Path | Contents |
|---|---|
| `PREREGISTRATION.md` | hypotheses, metrics and decision rules, frozen before any run |
| `SPRINT_LOG.md` | every deviation and correction, with evidence |
| `src/refusals_that_stick/` | survival metric, data build, serving, judging, repairs, statistics |
| `scripts/` | analyses, figures, judge scoring, prior-work scan, release |
| `results/` | every result file behind every number in this README |
| `docs/figures/` | the figures above |
| `tests/` | 174 CPU tests |
| `index.html` | the project website |

---

## Limitations

- **One model and one drift.** Results are for Tiny Aya Fire with a 3% harmful dose. A milder
  drift could leave room for a repair that passes the gate.
- **Partial repairs.** Both repairs restore only 24–36% of the lost refusal, so the study measures
  how much of a partial repair survives.
- **Behavioural power.** The test split is the largest available for Hindi, and it cannot resolve
  a loss of the estimated size.
- **Judge.** Refusal is judged by one model, validated against 150 labels from a single labeller,
  the author.
- **The fix is measured in the weights, not yet in behaviour.** Edit-aware protection restores
  weight-level survival, but no test of feasible size can show its behavioural effect. The
  importance-matrix fix (H5) was screened only on Fire at Q2_K, on the dev split.

## With more credits

What I would run next, in order, and the result each would settle:

1. **A test large enough to see the loss.** About 3,500 red-team prompts per language would
   resolve the sparse repair's estimated 2.3-point loss at Q4_K_M. That needs Hindi red-team
   prompts beyond the 913 Aya ones, for example English red-team sets translated with the same
   back-translation check used for XSTest, then the four test cells run once.
2. **The protected file on that test.** Keeping the 15 edited tensors at Q8_0 raises weight-level
   survival from 55.6% to 91.1%. Only a test of that size can show whether this reaches
   behaviour.
3. **A milder drift.** A 1% harmful dose, or the plain Hindi fine-tune, may leave room for a repair
   that passes the pre-registered gate, so the survival question can be asked of a full repair
   rather than a partial one.
4. **Train-time hardening.** SafeTune's train-time methods were outside this study; the same
   per-weight survival measurement applies to them.
5. **Other models.** The same protocol on Tiny Aya Global and Earth, and on Aya Expanse 8B.

## Corrections made during the project

Each is recorded in `SPRINT_LOG.md` with the evidence that caught it:

1. A decoding failure was first misdiagnosed as greedy repetition. The token ids showed it was the
   missing stop token.
2. The over-refusal limit was first measured against the drifted model. It was rebased to Fire
   before any repair result could depend on it.
3. The screening split suggested the sparse repair was undetectable at Q4_K_M. The held-out test
   split showed this was a power artifact, and the claim was withdrawn.
4. An early draft compared directional survival with 50%. Because erased weights count as
   failures, the correct chance level is half the share that moved. The text now compares
   "intended among moved" with 50%.

## Release and licence

All Hugging Face artifacts are public, not gated, and CC-BY-NC, inherited from Tiny Aya.

- **No model response text is published.** The drifted model complies with harmful requests, so
  only labels and metadata are released. The one exception is `results/diag_decoding.json`, which
  keeps a few short MMLU answers from the unmodified Fire model as evidence for the stop-token bug.
- **The harmful dose is identified by BeaverTails row index**, not redistributed.
- **The drifted model is released as a LoRA adapter only**, with a warning card. No merged unsafe
  checkpoint is published.
- Lexsi's CuratorKIT, SafeTune, CircuitKIT and AuditKIT are used under their licences and are not
  redistributed.

Everything in this repository was built during the hackathon.
