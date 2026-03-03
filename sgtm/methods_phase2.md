# Phase 2 Methods: Planned Experiments

## Overview

Phase 2 runs corrected SGTM experiments addressing the limitations identified in Phase 1 (see `methods_phase1.md`). The primary experiment tests a coarse forget task (all viral vs non-viral) with reduced forget allocation matching the original SGTM paper's ~3% target. The secondary experiment repeats the fine-grained task (human-infecting vs other viral) and tests recovery fine-tuning of the holdout model.

The goal is to determine whether Phase 1's catastrophic ablation failure was caused by (a) over-allocation of forget capacity, (b) the difficulty of the fine-grained task, (c) the four-way data split, or (d) a fundamental incompatibility between SGTM and masked protein language models.

## Changes from Phase 1

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Forget allocation | 10–15% of heads, 12.5% of MLP | ~5% attn (1 head), ~3% MLP |
| Data categories | 4 (forget/adjacent/ambiguous/retain) | 2 (coarse) or 3 (fine) |
| Forget task | Human-infecting viral vs non-human viral | Both coarse (all viral) AND fine (human-infecting only) |
| Ablation conditions | Attn-only + Attn+MLP | Attn+MLP only (matches paper) |
| Shared biases | Not partitioned | Still not partitioned (noted as limitation) |
| Data provenance | Undocumented | Reproducible download script |

## Model Architecture

We use the same ESM-2 architecture at 8M and 35M scales as Phase 1 (see `methods_phase1.md` for the architecture table).

The key architectural change is reduced forget allocation:

| | ESM-2 8M | ESM-2 35M |
|---|---|---|
| Forget heads | [19] (1/20 = 5%) | [19] (1/20 = 5%) |
| Forget MLP start | 1,240 (40/1,280 = 3.1%) | 1,860 (60/1,920 = 3.1%) |
| Total forget % (approx.) | ~4% | ~4% |

These defaults are defined in `sgtm/model_config.py` and can be overridden at the command line with `--forget-heads` and `--forget-mlp-start`.

**[DECISION NEEDED] 150M scale:** Add ESM-2 150M to the config registry? This enters the bracket of the original SGTM paper (GPT-2 117M). Estimated cost ~$88–264 depending on conditions and mixed precision. Budget would need to cover at minimum holdout + 1 SGTM condition.

## Data

### Reproducible Download

All data is generated via `data/download_virus_data.py` for curated viral datasets and `sgtm/data_pipeline.py` for the full pipeline (Swiss-Prot download, filtering, splitting). The pipeline is run as:

```
python -m sgtm.data_pipeline --forget-task {coarse,fine,custom}
```

Data is saved to `data/sgtm/{coarse,fine,custom}/` as HuggingFace datasets with a JSON manifest documenting the configuration.

### Coarse Task (`--forget-task coarse`)

A clean two-way split with no adjacent category:

| Category | Source | SGTM Mode |
|---|---|---|
| **Forget** | All viral proteins (curated human + curated non-human + Swiss-Prot header-detected) | Forget |
| **Retain** | Non-viral Swiss-Prot proteins | Retain |

This is the primary experiment. If SGTM cannot separate "all viral" from "non-viral" — categories that are structurally distinct — then it cannot do anything useful on protein models.

### Fine Task (`--forget-task fine`)

A three-way split matching the original SGTM paper's design:

| Category | Source | SGTM Mode |
|---|---|---|
| **Forget** | Human-infecting viral proteins (curated, species-level annotation) | Forget |
| **Adjacent** | Non-human viral (curated) + Swiss-Prot viral (header-detected) | Default |
| **Retain** | Non-viral Swiss-Prot proteins | Retain |

This merges Phase 1's "adjacent" and "ambiguous" categories into a single "adjacent" category, eliminating the redundant four-way split. The fine task is harder and closer to the real biosecurity use case, but may require larger models.

### Custom Task (`--forget-task custom --forget-tsv PATH`)

User-specified forget set from a TSV file. Adjacent = remaining viral proteins not in the forget set. Retain = non-viral Swiss-Prot.

### Split Ratios

Same as Phase 1:
- Forget and adjacent: 90/5/5 (train/val/test)
- Retain: 95/2.5/2.5

## SGTM Implementation

### Parameter Partitioning and Gradient Masking

Same partitioning and masking approach as Phase 1 (see `methods_phase1.md` for full details). The changes are:

1. **Only attn+MLP condition** — no attention-only condition (matches the original paper).
2. **1 head + small MLP slice** — ~5% attention + ~3% MLP, totaling ~4% of each layer's parameters (vs Phase 1's ~11–13%).

### Gradient Masking Table

**Coarse task (2-way):**

| Data Category | SGTM Mode | Gradient Behavior |
|---|---|---|
| Forget (all viral) | "forget" | Zero gradients on retain parameters |
| Retain (non-viral) | "retain" | Zero gradients on forget parameters |

**Fine task (3-way):**

| Data Category | SGTM Mode | Gradient Behavior |
|---|---|---|
| Forget (human-infecting) | "forget" | Zero gradients on retain parameters |
| Adjacent (other viral) | "default" | No masking (all parameters update) |
| Retain (non-viral) | "retain" | Zero gradients on forget parameters |

### Ablation

Same as Phase 1: post-training zeroing of forget parameters, no rescaling.

## Training Conditions

### Primary: Coarse Task

| Condition | Training Data | Gradient Masking |
|---|---|---|
| **Holdout** | Retain only (no viral data) | None |
| **SGTM Attn+MLP** | Forget + Retain | Head [19] + MLP [1860:1920] (35M) or [1240:1280] (8M) |

### Secondary: Fine Task

| Condition | Training Data | Gradient Masking |
|---|---|---|
| **Holdout** | Adjacent + Retain (no forget) | None |
| **SGTM Attn+MLP** | Forget + Adjacent + Retain | Head [19] + MLP [1860:1920] (35M) or [1240:1280] (8M) |

### Shared Hyperparameters

Same as Phase 1:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 0.01 |
| LR schedule | Linear warmup (2,000 steps) + cosine decay |
| Training steps | 40,000 |
| Batch size (physical) | 32 |
| Gradient accumulation | 4 (effective batch = 128) |
| Max gradient norm | 1.0 |
| Sequence max length | 1,022 tokens |
| MLM mask ratio | 15% |
| Random seed | 42 |

## Evaluation

### Deterministic Eval Masking

Pre- and post-ablation PPL evaluations use a dedicated `torch.Generator` inside `MLMCollator(seed=42)`, isolating eval masking from the global RNG. The collator's generator is reset with `collator.rng.manual_seed(42)` before each evaluation pass, guaranteeing identical masked positions regardless of other operations. This replaces Phase 1's fragile global seed resets.

### Pseudo-Perplexity (PPL)

Same definition as Phase 1: exponential of the average cross-entropy loss on masked tokens, computed on held-out test sets for each data category.

### Ablation Ratios

The primary cross-architecture metric is the **ablation ratio**: PPL_post / PPL_pre per split. This is a dimensionless ratio comparable across MLM and autoregressive models (where absolute PPL values are fundamentally different). Ablation ratios are computed and stored in the results JSON by `evaluate_sgtm.py`.

Goal: forget ablation ratio >> 1 (viral knowledge removed), retain ablation ratio ≈ 1 (general capability preserved).

### Linear Probes (Corrected)

Linear probes use the corrected methodology:

- **Val splits only** — `load_virus_sequences()` and `load_viral_vs_nonviral()` load from `data/sgtm/{split}/val`, not raw TSVs
- **Balanced classes** — Task 2 downsamples the majority class to match the minority class size
- **Balanced accuracy** — scoring uses `sklearn.metrics.balanced_accuracy_score` instead of standard accuracy
- 5-fold cross-validation with mean and standard deviation reported

### Recovery Fine-Tuning

The holdout model (trained without forget data) is fine-tuned on the forget set using the same MLM objective, simulating an adversary who obtains the filtered model and the filtered-out data.

| Parameter | Value |
|---|---|
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 16 |
| Max steps | 2,000 |
| Eval interval | Every 50 steps |
| Optimizer | AdamW (weight decay 0.01) |
| LR schedule | Cosine decay (no warmup) |

Metrics:
- **Steps to recover:** Steps until forget PPL matches or exceeds the baseline (a model trained on all data)
- **Retain degradation:** Change in retain PPL during recovery
- **Adjacent transfer:** Whether fine-tuning on human-infecting viral proteins also improves performance on non-human viral proteins

Script: `sgtm/recovery_finetune.py`

## Open Decisions

The following design choices have not been finalized and require discussion before experiments begin.

### [DECISION NEEDED] Signal Matching

MLM predicts ~15% of tokens per step vs 100% for autoregressive models. At 40K steps, ESM-2 gets ~6K effective token predictions per position vs 40K for GPT-Neo. A "signal-matched" condition would need ~267K steps (6.7× compute). This is a real confound for cross-architecture interpretation: if SGTM fails on ESM-2, it could be undertrained rather than architecturally incompatible.

Options:
- Run one signal-matched condition at 35M (6.7× budget for that condition)
- Note as a limitation and defer
- The parallel BERT-SGTM experiment runs both step-matched and signal-matched, which may inform this decision

### [DECISION NEEDED] Multiple Seeds

All Phase 1 experiments used seed=42 with no variance reporting. Running 3 seeds for the SGTM conditions would 3× the SGTM training budget.

Options:
- Run 3 seeds from the start (~3× SGTM budget)
- Run 1 seed first; if a condition shows promising results, rerun with 2 additional seeds before claiming success
- Note as a limitation and report single-seed results

### [DECISION NEEDED] Weight Decay

We use `weight_decay=0.01` (ESM-2 convention) vs the original SGTM paper's GPT-Neo experiments using `0.1`. Higher weight decay encourages sparser representations, potentially making SGTM's localization task easier.

Options:
- Keep 0.01 (defensible as ESM-2's convention)
- Add one `weight_decay=0.1` condition (matches GPT-Neo, enables cleaner cross-architecture comparison)
- Note as a limitation

### [DECISION NEEDED] 150M Scale

ESM-2 150M would enter the bracket of the original SGTM paper's smallest successful model (GPT-2 117M). Estimated cost ~$88–264.

Options:
- Add 150M config to `model_config.py` (needs: num_layers, embed_dim, attention_heads, head_dim, mlp_dim)
- Run at minimum holdout + 1 SGTM condition at 150M
- Defer until 35M Phase 2 results are in

### [DECISION NEEDED] Autoregressive Protein Models

Testing SGTM on an autoregressive protein model would clarify whether Phase 1 results are encoder-specific. Candidates:
- **RITA Small (85M):** GPT-style, protein sequences
- **ProGen2-Small (151M):** Autoregressive, closest to SGTM paper's architecture class
- **GPT-Neo path:** Use the SGTM repo's own GPT-Neo architecture with a protein tokenizer

Options:
- Pursue alongside 150M ESM-2
- Defer as a follow-up after Phase 2 ESM-2 results

## Contingent Experiments

The following experiments are planned only if the primary coarse SGTM experiment succeeds (retain PPL preserved, forget PPL meaningfully increases after ablation).

### Absorption Test

The original SGTM paper highlights absorption as a key mechanism: ambiguous or borderline data, trained with `default` mode (all parameters update), gets absorbed into the retain pathway because retain parameters have greater capacity. The clean 2-way coarse experiment has no ambiguous data — every protein is unambiguously viral or non-viral — so it cannot demonstrate this effect.

If the coarse experiment works, we test absorption by randomly routing a fraction of the viral (forget) set through `default` mode instead of `forget` mode. For example, hold out 10–20% of viral proteins and train them with all parameters updating. If absorption works, these proteins' knowledge should end up primarily in retain parameters, and ablation should still increase forget PPL on the held-out fraction. This tests whether SGTM can handle imperfect labels — a practical requirement for real deployment where the boundary between dangerous and benign knowledge is fuzzy.

### Recovery Fine-Tuning (on SGTM-ablated model)

If coarse ablation produces a model with preserved retain PPL and degraded forget PPL, test whether an adversary can recover viral capability by fine-tuning the ablated model on viral data. Compare recovery speed to the holdout recovery experiment (which tests data filtering tamper-resistance).

## Known Limitations (Not Fixing)

The following limitations are acknowledged but will not be addressed in Phase 2:

1. **Shared biases still unpartitioned.** `out_proj.bias` and `fc2.bias` are not partitioned. The original SGTM implementation handles biases differently. We note this as a deviation but do not change it to keep the codebase consistent across phases.

2. **Uniform masking across layers.** All layers use the same forget head indices and MLP split. A non-uniform strategy (e.g., targeting specific layers where viral knowledge localizes) might improve separation, but this is not explored.

3. **Training from scratch.** We train from random initialization, not from pretrained ESM-2 weights. This tests SGTM in the from-scratch training regime (where it was designed to operate) but means our models are weaker than pretrained ESM-2 at the same parameter count.

4. **Proxy forget set.** Our forget set (human-infecting viral proteins, or all viral proteins) is a proxy for genuinely dangerous biological knowledge. A real deployment scenario would involve more specific threat-relevant sequences (e.g., gain-of-function mutations, select agents). The proxy is sufficient for testing SGTM mechanics but not for assessing real-world biosecurity implications.
