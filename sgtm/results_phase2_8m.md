# Phase 2 Preliminary Results: 8M Coarse Task

**Date:** 2026-03-06
**Model:** ESM-2 8M (7.5M parameters, 6 layers, 20 heads, 320-dim)
**Task:** Coarse forget (all viral vs non-viral)
**Data:** 14,384 forget train / 424,428 retain train (~3.4% forget steps)

## Summary

SGTM gradient routing does not achieve knowledge localization at 8M scale. Ablation is catastrophic across all conditions — zeroing the forget partition (~5% of attention, ~3% of MLP) destroys both forget and retain capability equally. The retain mode split is a powerful lever (20× improvement from ret10 to ret25) but insufficient to make ablation non-destructive at this model size.

## Implementation Fixes (vs Phase 1)

Three bugs were found by ML review against the original SGTM paper's code and fixed before these experiments:

1. **Gradient masking timing:** Phase 1 applied `adjust_gradients()` per micro-batch inside the accumulation loop. Only the last micro-batch's gradients survived on masked parameters, giving them 4× noisier effective training. Fixed: masking applied once after all micro-batches accumulate.

2. **Incomplete embedding masking:** Phase 1 `--mask-embeddings` only zeroed token embedding gradients. The paper also zeros all layer norm gradients in forget mode. Fixed: added layer norm gradient zeroing for `self_attn_layer_norm`, `final_layer_norm`, and `emb_layer_norm_after`.

3. **Retain mode split:** Phase 1 used 100% retain mode (forget params only received gradient on ~4% of steps). The paper uses probabilistic assignment: 10-25% retain mode, 75-90% default mode. Fixed: added `--retain-retain-perc` flag.

## Conditions

| Condition | Retain Mode Split | Embedding Masking | Other |
|---|---|---|---|
| Holdout | N/A | N/A | Trained on retain data only |
| SGTM ret10 | 10% retain / 90% default | Yes | Matches paper's 254M config |
| SGTM ret25 | 25% retain / 75% default | Yes | Matches paper's 8M TinyStories config |

All conditions: 40K steps, effective batch 128 (4×32), LR 5e-4, weight decay 0.01, cosine schedule with 2K warmup, seed 42, no upsampling.

## Perplexity Results (Test Set)

| Condition | Forget PPL | Retain PPL | Forget Ablation Ratio | Retain Ablation Ratio |
|---|---|---|---|---|
| Holdout | 14.04 | 9.42 | — | — |
| SGTM ret10 | 13.99 | 9.77 | — | — |
| SGTM ret10 ablated | 4,079 | 9,417 | 292× | 964× |
| SGTM ret25 | 13.98 | 9.85 | — | — |
| SGTM ret25 ablated | 682 | 454 | 49× | 46× |

Goal: forget ratio >> 1, retain ratio ≈ 1.

## Linear Probe Results (ret10 only)

Balanced accuracy, 5-fold CV, viral vs non-viral classification:

| Condition | Balanced Accuracy |
|---|---|
| Holdout | 0.822 ± 0.013 |
| SGTM ret10 | 0.828 ± 0.008 |
| SGTM ret10 ablated | 0.732 ± 0.041 |

Probe accuracy drops modestly after ablation (0.828 → 0.732) even as generation PPL explodes (14 → 4,079). Viral knowledge persists in representations even when the parameters that use it for generation are zeroed.

## Key Observations

### 1. Retain mode split is a powerful lever

Ret25 reduces the retain ablation ratio by 20× compared to ret10 (46× vs 964×). Pre-ablation PPL is virtually identical between conditions. The split controls how dependent the model becomes on the forget partition, without affecting training quality.

### 2. No localization at 8M

For both ret10 and ret25, the forget and retain ablation ratios are similar (ret10: 292× vs 964×; ret25: 49× vs 46×). The forget partition is not preferentially storing forget-specific knowledge — it carries roughly equal importance for all tasks. SGTM's gradient routing is not producing the asymmetric localization needed for selective ablation.

### 3. Pre-ablation performance is unaffected by SGTM

All three conditions produce nearly identical pre-ablation PPL (forget: 13.98-14.04, retain: 9.42-9.85). The gradient routing neither helps nor hurts training — it only changes how the model distributes computation across partitions.

### 4. Representations survive ablation (generation doesn't)

The linear probe drops from 0.828 to 0.732 after ablation — a modest decrease. But PPL goes from 14 to 4,079. The model's internal representations still encode viral vs non-viral information, but the parameters needed to *use* that information for generation are destroyed. This is consistent with the SGTM paper's finding that representation-level knowledge is harder to ablate than generation-level capability.

## Interpretation

The 8M model has 7.5M parameters total. The forget partition (~5% attention + ~3% MLP) represents ~4% of each layer's trainable capacity. At this scale, every parameter is load-bearing — there is not enough redundancy for any partition to be "expendable." Zeroing 4% of the model's capacity causes roughly proportional damage to all capabilities, not selective damage to forget capability.

The paper's smallest model (GPT-Neo 8M for TinyStories) has 32 heads with head_dim=4 — many tiny heads where losing 1 is a small perturbation. Our ESM-2 8M has 20 heads with head_dim=16 — fewer, larger heads where losing 1 is a bigger structural change.

## Next Steps

1. **35M experiments** — more parameters = more redundancy. The forget partition becomes a smaller fraction of total capacity. Run holdout + SGTM ret25.
2. **If 35M also fails** — points to a fundamental issue with SGTM on masked protein language models rather than a scale problem. Consider 150M (which enters the paper's validated parameter range) or autoregressive protein models.
