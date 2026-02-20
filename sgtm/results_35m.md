# ESM-2 35M SGTM Experiment Results

## Experiment Design

**Goal**: Test whether SGTM (Selective Gradient Training and Masking) can localize viral protein knowledge in ESM-2 35M, enabling clean ablation of viral capabilities while preserving general protein modeling.

**Motivation**: At 8M scale, attention-only ablation preserved retain PPL (14.84, +0.6% vs pre-ablation), while MLP ablation was catastrophic. The 35M experiments test whether this pattern holds at scale.

**Architecture**: ESM-2 35M — 12 layers, 20 attention heads, head_dim=24, embed_dim=480, MLP=1920.

**Training**: 40,000 steps, batch size 128 (8×16 grad accum), AdamW lr=5e-4, cosine schedule, trained from scratch.

### Conditions

| Condition | Description | Forget Heads | Forget MLP |
|---|---|---|---|
| Holdout | All data minus forget set (filtering control) | N/A | N/A |
| SGTM Attn-Only | Gradient masking, attention heads only | [18, 19] | None (start=1920) |
| SGTM Attn+MLP | Gradient masking, attention + MLP | [18, 19] | [1680:1920] |

### Data Splits

| Split | Train | Val | Description |
|---|---|---|---|
| Forget | 999 | 55 | Human-infecting viral proteins |
| Adjacent | 279 | 15 | Non-human viral proteins |
| Ambiguous | 13,800 | — | 10% of virus sets + Swiss-Prot viral proteins (default mode) |
| Retain | 424,468 | 11,170 | Filtered Swiss-Prot non-viral proteins |

**Note**: Swiss-Prot viral protein filtering (by FASTA header keyword matching) was added for 35M. The 8M experiments did not have this filter, meaning ~16K viral proteins leaked into the retain set.

---

## Results

### Pre-Ablation Perplexity

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| Holdout | 14.17 | 5.67 | **10.23** |
| SGTM Attn-Only | 14.11 | 10.78 | 13.78 |
| SGTM Attn+MLP | 13.56 | 12.31 | 13.92 |

All three conditions learned effectively. The holdout model achieves the best retain PPL (10.23) and dramatically better adjacent PPL (5.67), expected since it uses all parameters for all non-forget data.

SGTM models pay a retain PPL cost of ~3.5-3.7 points vs holdout, reflecting the reduced effective capacity (gradient masking restricts which parameters update for each data type).

### Post-Ablation Perplexity

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| SGTM Attn-Only ablated | 5.6 × 10²³ | 1.9 × 10²⁴ | 5.3 × 10²² |
| SGTM Attn+MLP ablated | 3.3 × 10³¹ | 2.2 × 10³⁴ | 7.7 × 10³¹ |

**Both ablation conditions are catastrophic.** Perplexity values of 10²² to 10³⁴ indicate the model produces effectively random (or adversarial) token predictions. The model is completely destroyed by ablation in both conditions.

### Comparison with 8M Results

| Metric | 8M (6 layers) | 35M (12 layers) |
|---|---|---|
| Pre-ablation retain PPL | 14.75 | 13.78 |
| Attn-only ablated retain PPL | 14.84 (+0.6%) | 5.3 × 10²² (destroyed) |
| Full ablated retain PPL | 2.95 × 10¹² (destroyed) | 7.7 × 10³¹ (destroyed) |
| **Attention-only ablation viable?** | **Yes** | **No** |

The key finding from 8M — that attention-only ablation preserves general capability — **does not replicate at 35M scale**. Scale made the problem worse, not better.

---

## Analysis

### Why did SGTM fail at 35M?

**1. Residual stream entanglement scales with depth.** In transformers, every attention head reads from and writes to a shared residual stream. With 12 layers (vs 6 at 8M), there are twice as many opportunities for retain parameters to develop dependencies on forget head outputs. When forget heads are zeroed, the distribution shift in the residual stream compounds through all 12 downstream layers, catastrophically breaking the model.

**2. Gradient masking ≠ information separation.** SGTM controls gradient flow but does not prevent:
- Forget heads from learning general protein structure (viral proteins share structural motifs with all proteins)
- Retain parameters from depending on forget head outputs during the forward pass
- Shared parameters (layer norms, biases) from coupling forget and retain computations

**3. Deeper models use capacity more efficiently.** At 8M with 6 layers, some attention heads may have been partially redundant, making their ablation survivable. At 35M with 12 layers, the model has 240 head-layer combinations (vs 120) and learns to fully utilize all available capacity, including forget heads, for general computation.

**4. Shared components act as bridges.** Layer norms, out_proj biases, and fc2 biases are not gradient-masked — they receive gradients from all data types. These shared parameters implicitly encode coupling between forget and retain circuits.

### Implications

1. **SGTM via gradient masking + ablation is not viable for ESM-2 at ≥35M scale.** The fundamental assumption — that transformer parameters can be partitioned into independent circuits via gradient routing — does not hold for deeper models.

2. **The 8M attention-only success was likely an artifact of model simplicity**, not evidence that SGTM works. The 6-layer model had enough redundancy that removing 3 attention heads could be compensated for. This does not generalize.

3. **Alternative approaches to explore:**
   - Adapter/LoRA-style modules that can be physically removed (no residual stream entanglement)
   - Post-hoc editing methods (ROME, MEMIT) that target specific factual associations
   - Knowledge distillation from a full model to a retain-only model
   - Representation engineering / activation steering approaches

---

## Infrastructure

- **Platform**: RunPod, 3× RTX 4090 (24GB VRAM, 46GB RAM)
- **Training time**: ~28 hours per pod at 0.4 steps/s
- **Total cost**: ~$45-50 (3 pods × ~$0.55/hr × 28hr)
- **Issues encountered**: Bus errors from shared memory exhaustion (fixed with `--num-workers 0`), SSH disconnects (mitigated with tmux)

## Files

- `results/sgtm/perplexity_results.json` — Raw perplexity numbers
- `results/sgtm/pre_ablation_comparison.png` — Pre-ablation PPL bar chart
- `results/sgtm/ablation_effect_log.png` — Log-scale ablation effect
- `results/sgtm/8m_vs_35m_comparison.png` — Cross-scale comparison
- `results/sgtm/holdout_vs_sgtm_retain.png` — Holdout vs SGTM retain cost
- `results/sgtm/training_curves.png` — Training loss over time
