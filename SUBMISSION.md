# Refusals That Stick — submission

**Track 2b (AI Safety and Correction) · `CohereLabs/tiny-aya-fire` · solo entry, Shashank Kumar**

## Abstract

A developer fine-tunes an aligned model for their language, safety drifts, they repair the
weights with a published method, then quantize to GGUF for a phone. Is the repair still there?
MANSU (arXiv:2605.15138) proves 4-bit quantization can erase an *unlearning* edit and notes the
repair setting "does not require" that guarantee. C-ΔΘ's paper never mentions quantization; nor
does the card for Lexsi's shipped repair checkpoint.

Every number comes from real GGUF files, **bit-identical to Cohere's published ones** (290/290
tensors at Q8_0 and Q4_K_M): no simulation caveat. The judge was validated
against 150 hand labels — its refusal label passed the pre-registered bar (κ = 0.73), its
harmfulness label failed (κ = 0.52), so refusal is primary and harmfulness only caveated
secondary. Gibberish never counts as safe.

**Four published repair methods, 13 configurations, none usable.** Fine-tuning Fire on Hindi data
plus a 3% harmful dose drops Hindi valid refusal from 71% to 23%. Sweeping every data-free
SafeTune repair (C-ΔΘ, RESTA, LoX, SafeMerge), nothing sits in the middle: every configuration
restoring ≥50% of the drift costs 28–65 MMLU points and raises perplexity ≥17×; every one
preserving capability restores ≤24%. The pre-registered gate failed; we report that rather than
lower it.

**At 4 bits a sparse repair's weights move in a near-random direction.** On the two *usable*
partial repairs, sparse (SafeMerge, 8.2% of weights) and dense (RESTA, 97.3%), 98.7% of the sparse
edit changes value at Q4_K_M, so a per-weight "did it change?" check passes it, yet only **55.6%**
of those weights move the way the repair intended, against 50% for a coin flip; dense keeps
**74.1%**. At Q4_0, 45% of the sparse edit rounds straight back. Sparse keeps less than dense at
**6 of 6** precisions, as pre-registered.

**Behaviourally, a null — and a retraction.** On the 150-prompt screening split the sparse
repair's 4-bit advantage looked undetectable, and I drafted that as the headline. The one-time
**663-prompt** test split says otherwise: both repairs stay significantly better than the drifted
model at Q4_K_M (sparse Hindi +4.7 pt, p = 0.001). The remaining loss is not significant: sparse
loses 2.3 of its 6.9 points of Hindi benefit, 95% bootstrap CI [−1.4, +5.9].

**Limits, with a floor.** H6 (weight survival predicts behavioural retention, ρ > 0.5) is
**falsified**: ρ = 0.38. Resolving a third of a 6.9-point benefit needs ≈3,500 red-team prompts
per language, four times the whole Hindi Aya set, so neither level proxies the other.

**A fix.** Keeping the 15 tensors the sparse repair edited at Q8_0 inside the Q4_K_M file
(`llama-quantize --tensor-type`) raises intended-direction survival from 55.6% to **91.1%** for
**7.5%** more file size, against 66.9% for the whole model at Q8_0, verified on a built file.
The pre-registered circuit-guided variant beats its random control only weakly.

**Side finding:** Fire's GGUF — including Cohere's official file — omits the model's own stop
token, so llama.cpp generates past it, corrupting 49–55% of short answers until fixed.

## Links

- Code: https://github.com/sidthebuilder/refusals-that-stick
- Data: https://huggingface.co/datasets/Sha09090/refusals-that-stick-data
- Per-prompt results: https://huggingface.co/datasets/Sha09090/refusals-that-stick-results
- Drift adapter (warning card): https://huggingface.co/Sha09090/tiny-aya-fire-hindi-drift-lora

## Reproducibility

Plan frozen before any run (`PREREGISTRATION.md`); every deviation and every self-correction
logged with evidence (`SPRINT_LOG.md`), including four of my own errors, each corrected: a
decoding misdiagnosis, an over-refusal baseline, a dev-split headline and a chance baseline.
174 CPU tests, one pinned llama.cpp commit. Harmful completions are never published.

Website: https://sidthebuilder.github.io/refusals-that-stick/
