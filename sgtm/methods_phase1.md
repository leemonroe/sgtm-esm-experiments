# Phase 1 Methods: What Was Done

## Overview

We adapted Selective Gradient Training and Masking (SGTM; Anthropic, 2025) to ESM-2, a masked protein language model, and trained models from scratch at 8M and 35M parameter scales. At 35M, we tested three conditions: data filtering, SGTM with attention-only masking, and SGTM with attention + MLP masking. Post-training ablation of the designated forget parameters was catastrophic at 35M — perplexity jumped to 10²²–10³⁴ across all splits, destroying the model entirely. An attention-only ablation strategy that appeared viable at 8M (+0.6% retain PPL) did not replicate at 35M. These experiments produced a clear negative result: SGTM gradient masking, as configured, fails on ESM-2 at 35M scale.

## Model Architecture

We used the ESM-2 architecture (Lin et al., 2023), a BERT-style masked language model for protein sequences. ESM-2 takes a protein sequence as input, masks a subset of amino acid tokens, and is trained to predict the masked residues. The model produces per-token logits over a 33-token vocabulary (20 standard amino acids plus special tokens) and dense per-token embeddings. ESM-2 is an encoder model — it produces representations and per-position predictions, not generated sequences.

We trained two model sizes from random initialization (not from pretrained weights):

| | ESM-2 8M | ESM-2 35M |
|---|---|---|
| Layers | 6 | 12 |
| Attention heads | 20 | 20 |
| Head dimension | 16 | 24 |
| Embedding dimension | 320 | 480 |
| MLP intermediate dimension | 1,280 | 1,920 |
| Total parameters | ~8M | ~35M |

Models were initialized with PyTorch default initialization and trained from scratch to isolate the effect of SGTM from pretrained representations.

## Data

### Source

Viral protein sequences were drawn from two curated datasets:

- **Human-infecting viral proteins** (1,239 sequences): Proteins from viruses with documented human tropism, sourced from UniProt/Swiss-Prot. The human/non-human split was performed at the virus *species* level, not by the `Virus hosts` annotation field. As a result, 275 of 1,239 entries (22%) were avian/swine-only Influenza A strains included because Influenza A as a species infects humans, even though these specific strains do not. The file was 77% Influenza A (955/1,239), with the remainder being HIV, SARS-CoV-2, and other human-infecting viruses.
- **Non-human viral proteins** (349 sequences): Proteins from viruses not known to infect humans, 67% bacteriophage T4 (235/349) and 13% plant viruses.

**Data provenance gap:** There was no download script, no query documentation, and no record of the exact UniProt selection criteria used to produce these files. The TSVs appeared in the initial git commit as pre-existing artifacts. The provenance was reconstructed post-hoc by analyzing the files' content (UniProt accession IDs, `Virus hosts` field with NCBI TaxIDs).

General protein data was drawn from Swiss-Prot (UniProt Consortium). Sequences were filtered to contain only the 20 standard amino acids and fall within 30–1,022 residues in length. Duplicates against the viral datasets were removed.

### Viral Contamination Filter

Swiss-Prot contains approximately 16,000 viral protein entries beyond the curated viral datasets. In the 8M experiments, these were **not** filtered, allowing viral proteins to contaminate the retain set. For the 35M experiments, we implemented header-based keyword filtering: sequences whose UniProt `OS=` (organism) field contained "virus," "phage," "viridae," "viral," or "virales" were removed from the retain set and routed to the ambiguous category. This was a known methodological improvement between scales.

### Data Splits

Data was partitioned into four categories that mapped to SGTM training modes:

| Category | Source | Size (train) | SGTM Mode | Purpose |
|---|---|---|---|---|
| **Forget** | 90% of human-infecting viral | 999 | Forget | Targeted dangerous knowledge |
| **Adjacent** | 90% of non-human viral | 279 | Default | Generalization control — non-targeted viral |
| **Ambiguous** | 10% of both viral sets + Swiss-Prot viral proteins | ~13,800 | Default | Borderline viral data, all parameters update |
| **Retain** | Filtered Swiss-Prot non-viral | ~403K (8M) / ~424K (35M) | Retain | General protein knowledge |

Each category except ambiguous was further split into train/val/test:
- Forget and adjacent: 90/5/5
- Retain: 95/2.5/2.5
- Ambiguous: no split (used in training only)

The remaining 10% of each viral dataset (not assigned to forget or adjacent) was pooled into the ambiguous set.

### Upsampling

The viral categories were a small fraction of the total training corpus. We upsampled forget sequences by 50× and adjacent sequences by 150× when constructing the per-step data schedule. The full upsampled schedule contained ~509,000 entries, which was shuffled with a fixed random seed and truncated to the training length of 40,000 steps. Because the schedule was shuffled before truncation, the category proportions were approximately preserved:

| Category | Upsampled count | Proportion | Steps seen (of 40K) |
|---|---|---|---|
| Forget | 49,950 | 9.8% | ~3,977 |
| Adjacent | 41,850 | 8.2% | ~3,314 |
| Ambiguous | 13,800 | 2.7% | ~1,125 |
| Retain | 403,000 | 79.2% | ~31,584 |

### MLM Masking

We used BERT-style masked language modeling. At each training step, 15% of non-special tokens were selected for prediction: 80% were replaced with `[MASK]`, 10% with a random amino acid token, and 10% were left unchanged. The loss was computed only on selected positions using cross-entropy with `ignore_index=-100` for non-selected tokens.

## SGTM Implementation

### Parameter Partitioning

SGTM partitioned each layer's parameters into "forget" and "retain" regions using contiguous index ranges within the weight matrices. Two hyperparameters defined the partition:

- **Forget heads**: A set of attention head indices whose corresponding rows/columns in Q, K, V, and output projection matrices were designated as forget parameters.
- **Forget MLP start**: An index within the MLP intermediate dimension; all dimensions from this index onward were designated as forget parameters.

| Weight Matrix | Forget Region |
|---|---|
| Q, K, V projection weights | Rows [h_start · d_h, (h_end+1) · d_h) |
| Q, K, V projection biases | Indices [h_start · d_h, (h_end+1) · d_h) |
| Output projection weight | Columns [h_start · d_h, (h_end+1) · d_h) |
| fc1 (MLP up-projection) weight | Rows [forget_mlp_start, d_mlp) |
| fc1 bias | Indices [forget_mlp_start, d_mlp) |
| fc2 (MLP down-projection) weight | Columns [forget_mlp_start, d_mlp) |

The following parameters were **not** partitioned and were shared across both modes: output projection bias, fc2 bias, and all layer normalization parameters. The original SGTM implementation partitions biases as well — our simplification meant shared biases were updated by both forget and retain data.

The masks were applied **identically across all layers** — every layer used the same forget head indices and MLP split.

### Gradient Masking

During training, each batch was drawn from one data category, and the SGTM mode was set accordingly:

| Data Category | SGTM Mode | Gradient Behavior |
|---|---|---|
| Forget | "forget" | Zero gradients on **retain** parameters (only forget params update) |
| Retain | "retain" | Zero gradients on **forget** parameters (only retain params update) |
| Adjacent | "default" | No masking (all parameters update) |
| Ambiguous | "default" | No masking (all parameters update) |

Gradient masking was implemented via post-backward zeroing: after each micro-batch backward pass, gradients on the masked parameter region were set to zero before the optimizer step. Gradient accumulation used 4 micro-batches per optimizer step (effective batch size = 4 × 32 = 128). Gradient masking was applied after each micro-batch backward pass, before accumulation.

### Ablation

Post-training, forget parameters were ablated by setting their values to zero. This was irreversible zeroing — the forget parameter values were replaced with exact zeros. No rescaling or normalization adjustment was applied to the remaining parameters.

## Training Conditions

### 35M Experiments (Primary)

| Condition | Training Data | Gradient Masking | Forget Heads | Forget MLP Start |
|---|---|---|---|---|
| **Data filtered** | Adjacent + Ambiguous + Retain (no forget) | None | N/A | N/A |
| **SGTM Attn-Only** | All four categories | Heads 18–19 masked | 18, 19 | 1,920 (no MLP masking) |
| **SGTM Attn+MLP** | All four categories | Heads 18–19 + MLP masked | 18, 19 | 1,680 |

For SGTM Attn-Only, `forget_mlp_start` was set equal to the MLP dimension (1,920), meaning no MLP parameters were in the forget partition. For SGTM Attn+MLP, the top 240 of 1,920 MLP dimensions (12.5%) were designated as forget parameters.

The forget attention heads (18–19 of 20 total) represented 10% of total attention capacity. Combined with MLP masking in the Attn+MLP condition, approximately 12.5% of each layer's MLP capacity was additionally designated as forget.


### 8M Experiments (Prior)

The 8M experiments used 5 conditions. The SGTM conditions used 3 forget heads (17, 18, 19 of 20) with `forget_mlp_start=1120` (retaining 1,120 of 1,280 MLP dims = 12.5% forget MLP). An attention-only condition tested heads 17–19 with no MLP masking. The 8M experiments did not include the Swiss-Prot viral contamination filter.

### Shared Hyperparameters

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

### Pseudo-Perplexity (PPL)

We evaluated masked language modeling performance using pseudo-perplexity: the exponential of the average cross-entropy loss on masked tokens, computed on held-out validation/test sets for each data category (forget, adjacent, retain). Lower PPL indicates better prediction of masked amino acid identities.

PPL was computed both **pre-ablation** (all parameters intact) and **post-ablation** (forget parameters zeroed). The comparison between pre- and post-ablation PPL on the retain set was the primary metric for SGTM viability.

**Masking determinism:** Evaluation masking used a global `torch.manual_seed(42)` reset before each evaluation pass. This was fragile — any operation consuming global RNG state between resets could silently change the masks. Pre- and post-ablation evaluations may not have used identical masked positions if intermediate operations consumed RNG state. (This was identified post-hoc and fixed in Phase 2 code with a dedicated `torch.Generator`.)

### Perplexity Results

**35M Pre-Ablation:**

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| Data filtered | 14.17 | 5.67 | **10.23** |
| SGTM Attn-Only | 14.11 | 10.78 | 13.78 |
| SGTM Attn+MLP | 13.56 | 12.31 | 13.92 |

**35M Post-Ablation:**

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| SGTM Attn-Only ablated | 5.6 × 10²³ | 1.9 × 10²⁴ | 5.3 × 10²² |
| SGTM Attn+MLP ablated | 3.3 × 10³¹ | 2.2 × 10³⁴ | 7.7 × 10³¹ |

**8M Attention-Only Comparison:**

| Scale | Pre-ablation Retain PPL | Post-ablation Retain PPL | Change |
|---|---|---|---|
| 8M (6 layers) | 14.75 | 14.84 | +0.6% |
| 35M (12 layers) | 13.78 | 5.3 × 10²² | Destroyed |

### Linear Probe Classification

We trained logistic regression classifiers on mean-pooled last-layer embeddings to test whether internal representations encoded viral classification capability.

**Task 1 — Human-infecting vs. non-human viral:** Binary classification on embeddings of forget-set (human-infecting, label=1) and adjacent-set (non-human, label=0) sequences. 5-fold cross-validation.

**Task 2 — Viral vs. non-viral:** Binary classification on embeddings of viral (forget + adjacent, label=1) vs. non-viral (retain, label=0) sequences. 5-fold cross-validation.

**Methodological caveat:** The linear probe results reported below were evaluated on **training data, not held-out splits**, with unbalanced classes. Task 1 had a 78% majority-class baseline (1,239 vs 349), Task 2 had a 76% baseline (1,588 vs 500). Scoring used standard accuracy, not balanced accuracy. Post-ablation accuracies of 80–88% were therefore barely above chance. These results are **unreliable** and should be treated as preliminary until re-evaluated with corrected methodology (held-out val splits, balanced classes, balanced accuracy scoring). The corrected code exists but had not been re-run on GPU at the time of writing.

**Task 1 Results (35M, uncorrected methodology):**

| Condition | Accuracy | Std |
|---|---|---|
| Data filtered | 98.0% | 0.5% |
| SGTM Attn-Only | 94.6% | 1.9% |
| SGTM Attn-Only ablated | 88.2% | 1.7% |
| SGTM Attn+MLP | 85.5% | 10.3% |
| SGTM Attn+MLP ablated | 80.2% | 1.3% |

**Task 2 Results (35M, uncorrected methodology):**

| Condition | Accuracy | Std |
|---|---|---|
| Data filtered | 90.0% | 5.4% |
| SGTM Attn-Only | 80.0% | 12.9% |
| SGTM Attn-Only ablated | 81.3% | 5.8% |
| SGTM Attn+MLP | 83.7% | 4.8% |
| SGTM Attn+MLP ablated | 78.8% | 3.4% |

## Infrastructure

All training and evaluation was performed on RunPod cloud GPU instances with NVIDIA RTX 4090 GPUs (24GB VRAM). Each 35M training condition required approximately 28 hours on a single GPU. Total compute cost for the 35M experiments was approximately $45–50. Total project compute (8M + 35M) was approximately $90.

## Known Limitations

The following issues were identified during and after the experiments:

1. **Forget capacity over-allocation.** We allocated 10–15% of attention heads and 12.5% of MLP dimensions to the forget partition. The original SGTM paper allocated ~3.1% of parameters. Over-allocating forget capacity is known to hurt retain performance; the 3.5–3.7 point pre-ablation retain PPL penalty (vs data filtered) may partly reflect over-allocation rather than a fundamental property of protein models.

2. **Four-way data split with redundant ambiguous category.** The original SGTM paper uses three data categories (forget, adjacent, retain). Our pipeline had four: forget, adjacent, ambiguous, and retain. The distinction between "adjacent" (279 non-human viral from curated TSVs) and "ambiguous" (~13,800 Swiss-Prot viral) is an artifact of the data pipeline, not a meaningful conceptual distinction. Both train with `sgtm_mode="default"` (all parameters update). Routing ~13,800 ambiguous viral proteins through default mode meant all parameters — including retain parameters — learned from a large volume of viral data, potentially undermining SGTM's separation.

3. **Attention-only condition not in original paper.** The original SGTM paper always masks attention AND MLP together. Our "attention-only" condition was our own addition, motivated by the 8M result where it appeared viable (+0.6% retain PPL). At 35M it failed identically to attn+MLP.

4. **Fine-grained forget task may be too hard for small models.** The forget task asked the model to forget human-infecting viral proteins while retaining non-human viral proteins — a subtle biological distinction. A human-infecting influenza protein and a bird-infecting influenza protein are structurally very similar. Small models may not have enough capacity to develop separate representations for these categories.

5. **Data provenance undocumented; 22% mislabeling in forget set.** No download script existed at experiment time. 275 of 1,239 "human-infecting" entries were avian/swine Influenza A strains included at the species level, not the strain level. The forget set was effectively "forget Influenza A + HIV + SARS-CoV-2" rather than a principled biological boundary.

6. **Single seed, no variance reporting.** All experiments used seed 42. We did not report variance across random seeds for training runs — only for linear probe cross-validation folds.

7. **Evaluation masking used global RNG.** Pre- and post-ablation PPL evaluations relied on global `torch.manual_seed(42)` resets, which is fragile. Any operation consuming global RNG state between resets could silently change the masks, making pre/post-ablation comparisons unreliable.

8. **Weight decay 0.01 vs GPT-Neo's 0.1.** We used `weight_decay=0.01` (ESM-2 convention) versus the original SGTM paper's GPT-Neo experiments using 0.1. Higher weight decay encourages sparser representations, potentially making SGTM's localization task easier. This is a confound for any cross-architecture comparison.

9. **Linear probes evaluated on training data with class imbalance.** The original `linear_probe.py` loaded all raw TSV sequences (including training data) for evaluation, not held-out val splits. Combined with class imbalance (78% and 76% majority-class baselines), the reported probe accuracies are unreliable. Code was corrected to use val splits and balanced accuracy, but the corrected version had not been re-run on GPU.

10. **Viral contamination in 8M retain set.** The 8M experiments did not filter Swiss-Prot viral entries, allowing ~16,000 viral proteins to leak into the retain set. This was fixed for 35M via header keyword matching.

11. **Shared biases not partitioned.** `out_proj.bias` and `fc2.bias` were not partitioned, allowing information leakage between forget and retain pathways. The original SGTM implementation handles biases differently.

12. **Uniform masking across layers.** All layers used the same forget head indices and MLP split. Viral knowledge may localize to specific layers, and a non-uniform strategy could improve separation.

## Code Availability

All training, evaluation, and data processing code is available in the project repository:

- `sgtm/train_sgtm.py` — Training loop with gradient masking
- `sgtm/masking.py` — Parameter mask construction, gradient adjustment, and ablation
- `sgtm/data_pipeline.py` — Data filtering and splitting
- `sgtm/evaluate_sgtm.py` — PPL evaluation
- `sgtm/linear_probe.py` — Linear probe evaluation
- `sgtm/model_config.py` — Architecture configuration registry
