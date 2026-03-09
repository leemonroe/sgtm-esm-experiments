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

### Source Databases

All protein sequences come from two sources, used as-is with no manual inspection of individual entries:

1. **Swiss-Prot** (UniProt/Swiss-Prot): UniProt's reviewed partition, downloaded as `uniprot_sprot.fasta.gz` from the UniProt FTP server (`ftp.uniprot.org`). Contains ~570K protein sequences across all organisms. Each FASTA header includes an `OS=` field identifying the source organism. We use this file in bulk — no entries are individually verified.

2. **UniProt viral query** (UniProt REST API): A programmatic query (`taxonomy_id:10239 AND reviewed:true`) retrieving all reviewed entries classified under Viruses, with `virus_hosts` annotation fields. This provides the host-level metadata needed for the fine-grained forget task. Downloaded via `data/download_virus_data.py`.

### Viral vs Non-Viral Classification

The viral/non-viral boundary is defined entirely by UniProt's existing annotations — we do not independently verify whether sequences are correctly classified. Two automated methods are used:

- **API-queried set:** The UniProt REST API query returns all entries under taxonomy ID 10239 (Viruses). These are split into human-infecting vs non-human-infecting based on whether `Homo sapiens [TaxID: 9606]` appears in the `virus_hosts` field. **Known limitation:** host annotation is at the virus *species* level, not strain level — all Influenza A proteins are tagged as human-infecting even if the specific strain (e.g., A/Duck/England/1/1956 H11N6) is avian-only.

- **Swiss-Prot header keyword matching:** Additional viral proteins are identified by regex matching on the `OS=` organism field in Swiss-Prot FASTA headers. Keywords: "virus", "phage", "viridae", "viral", "virales" (case-insensitive). This is a heuristic — we have not validated its precision or recall against UniProt's taxonomic classification.

For the coarse task, both sources are merged into a single "all viral" forget set. Sequences appearing in both sources are deduplicated by exact sequence match.

### Sequence Filtering

All sequences are filtered before splitting:

| Filter | Criterion |
|---|---|
| Valid amino acids | Standard 20 AAs only (no X, U, B, Z, etc.) |
| Minimum length | 30 residues |
| Maximum length | 1,022 residues (ESM-2 context window minus special tokens) |
| Deduplication | Exact sequence match (within and across categories) |

### Reproducible Pipeline

The full pipeline is run as:

```
python data/download_virus_data.py --output-dir data/raw/
python -m sgtm.data_pipeline --forget-task {coarse,fine,custom}
```

Data is saved to `data/sgtm/{coarse,fine,custom}/` as HuggingFace Arrow datasets with a JSON manifest documenting the configuration. Scripts: `data/download_virus_data.py` (curated viral download), `sgtm/data_pipeline.py` (Swiss-Prot download, filtering, splitting).

### Coarse Task (`--forget-task coarse`)

A clean two-way split with no adjacent category:

| Category | Description | Source | Count (train) |
|---|---|---|---|
| **Forget** | All viral proteins | Curated human-infecting + curated non-human + Swiss-Prot header-detected | ~14,384 |
| **Retain** | Non-viral proteins | Swiss-Prot entries not matching viral keywords | ~424,428 |

Forget fraction: ~3.4% of training steps (no upsampling). This is the primary experiment. If SGTM cannot separate "all viral" from "non-viral" — categories that are structurally distinct — then it cannot do anything useful on protein models.

### Fine Task (`--forget-task fine`)

A three-way split matching the original SGTM paper's design:

| Category | Description | Source | Count (train) |
|---|---|---|---|
| **Forget** | Human-infecting viral proteins | Curated viral proteins from species with human hosts (TaxID 9606) | ~1,100 |
| **Adjacent** | Other viral proteins | Curated non-human viral + Swiss-Prot header-detected viral | ~13,300 |
| **Retain** | Non-viral proteins | Swiss-Prot entries not matching viral keywords | ~424,428 |

This merges Phase 1's "adjacent" and "ambiguous" categories into a single "adjacent" category, eliminating the redundant four-way split. The fine task is harder and closer to the real biosecurity use case, but may require larger models.

### Custom Task (`--forget-task custom --forget-tsv PATH`)

User-specified forget set from a TSV file. Adjacent = remaining viral proteins not in the forget set. Retain = non-viral Swiss-Prot.

### Split Ratios

Same as Phase 1:
- Forget and adjacent: 90/5/5 (train/val/test)
- Retain: 95/2.5/2.5

### Upsampling and Data Proportions                                                                                                                                          

**Decision: no upsampling (match the original SGTM paper).**                                                                                                                 
                                                        
Phase 1 upsampled forget sequences 50× and adjacent sequences 150× to ensure the model encountered viral data frequently. This was calibrated for a 999-sequence forget set.
With the coarse task's ~15,600 forget train sequences, the same 50× factor would produce ~780K forget entries vs ~383K retain — roughly 67% of training steps on forget data
after truncation to 40K steps. This would starve the retain set and likely degrade general protein modeling.

The original SGTM paper uses **no upsampling** (all upsample factors = 1.0). Their Wikipedia biology forget set was ~3.7% of total tokens, and training steps were allocated
proportionally to natural dataset sizes. Our coarse task produces a similar natural ratio: ~15,600 forget / ~383,000 retain ≈ 4% forget steps. We match the paper's approach.

**Tradeoff: natural proportions vs effective exposure.** The paper's 3.7% forget fraction was over a large corpus trained for many steps — each biology document was likely
seen multiple times. Our ~15,600 forget train sequences at ~1,600 forget steps (4% of 40K) means each forget sequence is seen roughly once. If SGTM requires multiple
exposures to localize knowledge into the forget partition, this could be insufficient. However, matching the paper's approach is the cleanest first experiment. If the coarse
experiment fails, insufficient forget signal is a natural follow-up hypothesis that can be tested by adding controlled upsampling (e.g., 5× to reach ~17% forget steps).

**Retain mode split.** The SGTM paper uses probabilistic mode assignment: on each retain-source step, a random draw determines whether to use `retain` mode (forget params masked) or `default` mode (all params update). The paper's primary experiments use 10% retain / 90% default for their 254M model and 25% retain / 75% default for their 8M model. This ensures forget parameters still receive general training signal from most of the data — without this, forget params are starved of gradient and become undertrained rather than specialized.

Our implementation matches this with `--retain-retain-perc` (default 10%). Phase 1 used 100% retain mode (equivalent to `--retain-retain-perc 100`), which was a major confound — forget params only received gradient on ~4% of steps (forget mode only). Phase 2 tests both ret10 (matching the paper's 254M config) and ret25 (matching the paper's 8M config) to determine the sensitivity of this parameter.

### Class Balance for Evaluation

The corrected linear probe evaluation (`sgtm/linear_probe.py`) addresses the Phase 1 class imbalance issues:

- **Task 1 (human-infecting vs non-human viral):** Only applicable to the fine task. Uses val splits from forget and adjacent categories.
- **Task 2 (viral vs non-viral):** Applicable to both tasks. Downsamples the majority class (retain) to match the minority class (forget, or forget + adjacent) before
evaluation.
- **Scoring:** Balanced accuracy (`sklearn.metrics.balanced_accuracy_score`) instead of standard accuracy, with 5-fold cross-validation.

For the coarse task, Task 1 is not meaningful (no adjacent category). Task 2 becomes the primary probe evaluation.


## SGTM Implementation

### Parameter Partitioning and Gradient Masking

Same partitioning and masking approach as Phase 1 (see `methods_phase1.md` for full details). The changes are:

1. **Only attn+MLP condition** — no attention-only condition (matches the original paper).
2. **1 head + small MLP slice** — ~5% attention + ~3% MLP, totaling ~4% of each layer's parameters (vs Phase 1's ~11–13%).
3. **Gradient masking applied once after accumulation** — Phase 1 applied `adjust_gradients()` inside the micro-batch loop, meaning only the last micro-batch's gradients survived on masked parameters. With `grad_accum=4`, masked params had 4× noisier effective training than unmasked params. Phase 2 applies masking once after all micro-batches accumulate, matching the original paper's `trainer.py`.

### Embedding and Layer Norm Masking

When `--mask-embeddings` is set (default for all Phase 2 SGTM runs), the following gradients are zeroed in forget mode after each optimizer step, preventing forget data from updating shared parameters:

- `embed_tokens.weight` — the token embedding matrix
- `self_attn_layer_norm.weight` and `.bias` — per-layer attention layer norm
- `final_layer_norm.weight` and `.bias` — per-layer FFN layer norm
- `emb_layer_norm_after.weight` and `.bias` — post-encoder layer norm

This matches the original paper's `--mask-embeddings` flag, which zeros both embeddings and all layer norms. ESM-2 has no position embeddings (it uses rotary), so only token embeddings are relevant.

### Gradient Masking Table

**Coarse task (2-way):**

| Data Category | SGTM Mode | Gradient Behavior |
|---|---|---|
| Forget (all viral) | "forget" (100% of forget steps) | Zero gradients on retain params; zero embedding + LN gradients |
| Retain (non-viral) | "retain" (N%) or "default" (100-N%) | retain: zero gradients on forget params. default: all params update |

Where N = `--retain-retain-perc` (10% or 25%, see conditions below).

**Fine task (3-way):**

| Data Category | SGTM Mode | Gradient Behavior |
|---|---|---|
| Forget (human-infecting) | "forget" (100%) | Zero gradients on retain params; zero embedding + LN gradients |
| Adjacent (other viral) | "retain" (N%) or "default" (100-N%) | retain: zero forget grads. default: all params update |
| Retain (non-viral) | "retain" (N%) or "default" (100-N%) | retain: zero forget grads. default: all params update |

### Ablation

Same as Phase 1: post-training zeroing of forget parameters, no rescaling.

## Training Conditions

### Primary: Coarse Task

| Condition | Training Data | Gradient Masking | Retain Mode Split |
|---|---|---|---|
| **Holdout** | Retain only (no viral data) | None | N/A |
| **SGTM ret10** | Forget + Retain | Head [19] + MLP [1240:1280] (8M) or [1860:1920] (35M) | 10% retain / 90% default |
| **SGTM ret25** | Forget + Retain | Same as ret10 | 25% retain / 75% default |

The ret10 condition matches the original paper's 254M configuration. The ret25 condition matches the paper's 8M TinyStories configuration and tests whether more aggressive masking of forget params during retain-source steps improves localization at small model sizes.

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
| Batch size (physical) | 4 |
| Gradient accumulation | 32 (effective batch = 128) |
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

### Label Noise Robustness Test

If the coarse experiment succeeds, we test SGTM's robustness to imperfect data labeling by randomly routing a fraction (10–20%) of the viral (forget) set through `default` mode instead of `forget` mode, simulating a classifier that misses some dangerous sequences.

If the model is robust to this noise, ablation should still increase forget PPL on the mislabeled fraction — their knowledge should end up primarily in retain parameters because retain parameters have greater capacity and receive gradient from default-mode steps.

**Note:** This tests label noise robustness, not "absorption" in the sense used by the SGTM paper. True absorption refers to genuinely ambiguous data that the model learns to route based on its content. Our test uses known-viral data with artificially randomized labels, which tests a different (but practically important) property: whether SGTM degrades gracefully when the forget/retain boundary is imperfect.

### Recovery Fine-Tuning (on SGTM-ablated model)

If coarse ablation produces a model with preserved retain PPL and degraded forget PPL, test whether an adversary can recover viral capability by fine-tuning the ablated model on viral data. Compare recovery speed to the holdout recovery experiment (which tests data filtering tamper-resistance).

## Known Limitations (Not Fixing)

The following limitations are acknowledged but will not be addressed in Phase 2:

1. **Shared biases still unpartitioned.** `out_proj.bias` and `fc2.bias` are not partitioned. The original SGTM implementation handles biases differently. We note this as a deviation but do not change it to keep the codebase consistent across phases.

2. **Uniform masking across layers.** All layers use the same forget head indices and MLP split. A non-uniform strategy (e.g., targeting specific layers where viral knowledge localizes) might improve separation, but this is not explored.

3. **Training from scratch.** We train from random initialization, not from pretrained ESM-2 weights. This tests SGTM in the from-scratch training regime (where it was designed to operate) but means our models are weaker than pretrained ESM-2 at the same parameter count.

4. **Proxy forget set.** Our forget set (human-infecting viral proteins, or all viral proteins) is a proxy for genuinely dangerous biological knowledge. A real deployment scenario would involve more specific threat-relevant sequences (e.g., gain-of-function mutations, select agents). The proxy is sufficient for testing SGTM mechanics but not for assessing real-world biosecurity implications.
