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

The two levels do not agree, and the study shows why: a behavioural evaluation of normal size
cannot resolve a weight-level loss of this size. Neither one can be used as a stand-in for the
other.

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

### 6. A control: quantization also degrades the aligned model

Fire itself, with no fine-tune and no repair, drops from 71.3% to 55.3% Hindi refusal at Q2_K, and
its Hindi perplexity rises from 8.40 to 12.02. This is why every repair is compared with the
drifted model *at the same precision*: it separates "quantization undid the repair" from
"quantization hurts this model generally".

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

---

## Pre-registered hypotheses

| | Hypothesis | Result |
|---|---|---|
| H0 | A sparse repair loses more of its edit under quantization than a dense one | **Supported** at 6 of 6 precisions¹ |
| H1 | Directional survival is below 0.9 at Q4_K_M | **Supported** (0.549 sparse, 0.738 dense) |
| H2 | Behavioural retention is lower in Hindi than in English | **Mixed** (10 of 12 dev cells; 1 of 2 test cells) |
| H3 | The sparse and dense repairs differ in behavioural retention | **Not established** (intervals overlap) |
| H4 | Circuit-guided mixed precision beats a random control at equal bytes | Not run |
| H5 | A safety-calibrated importance matrix beats a generic one | Not run |
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

---

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m pytest tests -q            # 165 tests, CPU only, no network or API keys
python scripts/make_figures.py       # rebuilds every figure from results/ (needs matplotlib)
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
| `tests/` | 165 CPU tests |

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
- **Fixes not tested.** The pre-registered mixed-precision and importance-matrix fixes (H4, H5)
  were not run.

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
