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
restoring ≥50% of the drift costs 26–65 MMLU points and raises perplexity ≥18×; every one
preserving capability restores ≤24%. The pre-registered gate failed; we report that rather than
lower it.

**4-bit quantization directionally randomizes a sparse repair.** On the two *usable* partial
repairs — sparse (SafeMerge, 8.2% of weights) and dense (RESTA, 97.3%) — per-weight direction
survival at Q4_K_M is **54.9% sparse against 73.8% dense**, where 50% is chance. At Q4_0 sparse
reaches **31.2%: worse than chance**, so deployed weights move *against* the repair.
Quantization does not shrink this edit, it scrambles its sign. Sparse is below dense at **6 of
6** precisions, as pre-registered. Yet 98.7% of those weights *do* move at Q4_K_M, so a magnitude
metric would call the repair preserved — which is why two such metrics were rejected on day zero.

**Behaviourally, a null — and a retraction.** On the 150-prompt screening split the sparse
repair's 4-bit advantage looked undetectable, and I drafted that as the headline. The one-time
**663-prompt** test split says otherwise: both repairs stay significantly better than the drifted
model at Q4_K_M (sparse Hindi +4.7 pt, p = 0.001). The dev result was a power artifact. The
remaining attenuation is not significant: per prompt, sparse loses 2.3 of its 6.9 points of Hindi
benefit, 95% bootstrap CI [−1.4, +5.9]. No interval excludes zero.

**Limits, with a floor.** H6 (weight metrics predict behavioural retention, ρ > 0.5) is
**falsified**: ρ = 0.38. The reason is measurable: resolving a third of a 6.9-point benefit needs
≈3,500 red-team prompts per language, four times the whole Hindi Aya red-team set. So: a measured
weight-level effect plus a behavioural null with a stated floor, not an erasure claim; neither
level proxies the other. Quantization degrades the *aligned* model too (71% → 55% at Q2_K), so
comparisons are against the drifted model at matched precision.

**Side finding:** Fire's GGUF — including Cohere's official file — omits the model's own stop
token, so llama.cpp generates past it, corrupting 49–55% of short answers until fixed.

## Links

- Code: https://github.com/sidthebuilder/refusals-that-stick
- Data: https://huggingface.co/datasets/Sha09090/refusals-that-stick-data
- Per-prompt results: https://huggingface.co/datasets/Sha09090/refusals-that-stick-results
- Drift adapter (warning card): https://huggingface.co/Sha09090/tiny-aya-fire-hindi-drift-lora

## Reproducibility

Plan frozen before any run (`PREREGISTRATION.md`); every deviation and every self-correction
logged with evidence (`SPRINT_LOG.md`), including three wrong calls I had to retract — two
diagnoses and one headline. 164 CPU tests. One pinned llama.cpp commit. Harmful completions are
never published.
