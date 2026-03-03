# Midpoint Report: Weight-Level Safeguards for Open-Weight Protein Language Models

**ERA AI x Biosecurity Fellowship — February 2025**

---

## 1. Introduction

The release of powerful protein language models as open weights creates a safeguard challenge with no clean analog in text-based AI. When a large language model is released openly, fine-tuning can strip away RLHF-trained refusal behavior — but the model at least had refusal behavior to strip. Protein language models have no equivalent. An embedding of a viral protein is not "helpful" or "harmful" in itself; it is a numerical representation that downstream users can apply to drug design or to pathogen engineering with equal facility. There is no "knowing versus helping" gap to exploit, no refusal behavior to train, and no generation step to steer. The model's dangerous capability *is* its general capability.

This makes weight-level interventions — modifications to the model's parameters that degrade dangerous capabilities before release — the only class of safeguard that survives open-weighting. If the weights are public, then access controls, output filtering, and inference-time monitoring are all moot. Whatever safety properties the model has must be baked into the weights themselves.

Selective Gradient Training and Masking (SGTM) is a recent technique developed at Anthropic (2025) that attempts exactly this. During training, gradient masking confines hazardous knowledge to a designated subset of parameters; post-training, those parameters are ablated (zeroed), surgically removing the dangerous capability while preserving general performance. SGTM has shown promising results on GPT-2 class language models (117M–774M parameters), but has never been tested on protein language models — a domain where the relationship between "dangerous" and "general" knowledge is fundamentally different from natural language.

This report presents the first empirical test of SGTM on protein language models. We applied SGTM to ESM-2 at two scales (8M and 35M parameters), targeting the model's ability to represent human-infecting viral proteins. The results are negative: SGTM fails catastrophically at 35M scale, and an attention-only ablation strategy that appeared viable at 8M does not generalize. We present these findings alongside linear probe analyses that reveal a deeper challenge — even models that never saw viral training data acquire viral classification capabilities, and models whose generation is destroyed by ablation retain viral knowledge in their representations. Together, these results have implications for the broader question of whether weight-level safeguards are viable for protein language models.

## 2. Background

### ESM-2 and Protein Language Models

ESM-2 (Lin et al., 2023) is a masked language model trained on protein sequences. Given a protein sequence with some amino acids masked, it predicts the identity of the masked residues — analogous to BERT's masked token prediction for text. The model learns rich representations of protein structure and function through this self-supervised objective, producing embeddings that transfer to downstream tasks like structure prediction, function annotation, and protein classification. Critically, ESM-2 is an *encoder* — it produces representations, not sequences. Its outputs are embeddings and per-token predictions, not generated proteins.

### Selective Gradient Training and Masking

SGTM (Anthropic, 2025) partitions model parameters into "forget" and "retain" sets. During training, gradient masking ensures that hazardous data (the "forget set") only updates forget parameters, while benign data (the "retain set") updates all parameters. This creates an asymmetry: forget parameters encode hazardous knowledge, but retain parameters learn general capabilities from all data. Post-training, zeroing the forget parameters should remove the hazardous knowledge while leaving general capabilities intact.

The technique was developed and validated on GPT-2 class autoregressive language models (117M–774M parameters), where it successfully removed knowledge of specific topics (e.g., a target language) while preserving general language modeling. The smallest model tested in the original work was GPT-2 Small at 117M parameters.

### The Knowing-Versus-Helping Gap

In text-based LLMs, there is a meaningful distinction between a model *knowing* something and *helping* a user do something. A model can represent knowledge about dangerous chemistry without producing synthesis instructions — this gap is what refusal training exploits. Protein encoder models have no such gap. A high-quality embedding of a pathogenic protein is simultaneously the knowledge and the help: the same representation that enables legitimate structural analysis enables malicious protein engineering. Any weight-level intervention must therefore operate on the knowledge itself, not on a behavioral layer sitting on top of it.

### Prior Work

The most directly relevant precedent is SGTM's application to GPT-Neo and GPT-2 models, where it achieved clean separation between forget and retain capabilities. In the protein domain, "Deep Ignorance" (2025) demonstrated that filtering pretraining data builds tamper-resistant safeguards into open-weight LLMs — the model never learns the dangerous knowledge, so adversarial fine-tuning cannot recover it. Our work tests whether a more targeted approach (SGTM) can achieve knowledge removal without requiring the model to be ignorant of the target domain entirely.

## 3. Experimental Setup

### Models

We trained ESM-2 models from scratch at two scales:

- **ESM-2 8M**: 6 layers, 20 attention heads, head_dim=16, embed_dim=320, MLP intermediate dim=1280
- **ESM-2 35M**: 12 layers, 20 attention heads, head_dim=24, embed_dim=480, MLP intermediate dim=1920

Both models were trained for 40,000 steps with batch size 128 (8 gradient accumulation steps × 16 physical batch), using AdamW with learning rate 5×10⁻⁴ and cosine scheduling.

### Data

| Split | Size (train) | Description |
|---|---|---|
| Forget | 999 | Human-infecting viral proteins |
| Adjacent | 279 | Non-human viral proteins |
| Ambiguous | ~13,800 | Swiss-Prot viral proteins (all parameters update) |
| Retain | 424,468 | Filtered Swiss-Prot non-viral proteins |

The forget set represents the "dangerous" knowledge we aim to remove — proteins from viruses known to infect humans. The adjacent set tests generalization to related but non-targeted viral proteins. The retain set is the general protein data the model should preserve capability on.

A key improvement from 8M to 35M: the 8M experiments did not filter Swiss-Prot entries by viral taxonomy, allowing approximately 16,000 viral proteins to leak into the retain set. The 35M experiments added FASTA header keyword matching to remove viral sequences from the retain split.

### Conditions

At 35M scale, we trained three conditions:

| Condition | Description | Forget Heads | Forget MLP |
|---|---|---|---|
| Holdout | Data filtering control — all data minus forget set | N/A | N/A |
| SGTM Attn-Only | Gradient masking on attention heads only | Heads 18–19 | None |
| SGTM Attn+MLP | Gradient masking on attention heads and MLP | Heads 18–19 | Dims 1680–1920 |

The holdout condition serves as a data filtering baseline — the simplest possible intervention. The two SGTM conditions test whether attention-only or attention+MLP ablation can achieve clean capability removal.

### Bug Fixes from 8M

Two bugs identified in the 8M experiments were corrected for 35M: (1) the viral contamination in the retain set described above, and (2) a hardcoded head_dim parameter that assumed 8M architecture dimensions. The 35M experiments use a config-driven architecture registry that parameterizes all dimensions.

### Infrastructure

All training was conducted on RunPod using RTX 4090 GPUs (24GB VRAM). Each condition trained on a single GPU for approximately 28 hours. Total compute cost for the 35M experiments was approximately $45–50 across three pods.

## 4. Results

### Pre-Ablation Perplexity

All three 35M conditions learned effectively during training:

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| Holdout | 14.17 | 5.67 | **10.23** |
| SGTM Attn-Only | 14.11 | 10.78 | 13.78 |
| SGTM Attn+MLP | 13.56 | 12.31 | 13.92 |

The holdout model achieves the best retain perplexity (10.23) and substantially better adjacent perplexity (5.67), as expected — it uses all parameters for all non-forget data. SGTM models pay a retain PPL cost of 3.5–3.7 points compared to the holdout, reflecting reduced effective capacity from gradient masking.

### Post-Ablation Perplexity

| Condition | Forget PPL | Adjacent PPL | Retain PPL |
|---|---|---|---|
| SGTM Attn-Only ablated | 5.6 × 10²³ | 1.9 × 10²⁴ | 5.3 × 10²² |
| SGTM Attn+MLP ablated | 3.3 × 10³¹ | 2.2 × 10³⁴ | 7.7 × 10³¹ |

**Both ablation conditions are catastrophic.** Perplexity values of 10²² to 10³⁴ indicate effectively random token predictions. Ablation does not selectively remove viral knowledge — it destroys the model entirely.

### 8M vs 35M: Attention-Only Ablation

The central question of the 35M experiment was whether the attention-only ablation result from 8M would replicate at scale:

| Scale | Pre-ablation Retain PPL | Post-ablation Retain PPL | Change |
|---|---|---|---|
| 8M (6 layers) | 14.75 | 14.84 | +0.6% (viable) |
| 35M (12 layers) | 13.78 | 5.3 × 10²² | Destroyed |

**It does not.** The attention-only ablation that appeared viable at 8M — a +0.6% perplexity cost — becomes catastrophic at 35M. Scale made the problem worse, not better.

### Linear Probe Analysis

Linear probes (logistic regression on mean-pooled last-layer embeddings) test whether the model's internal representations encode viral knowledge, independent of its token-prediction capability.

**Task 1: Human-infecting vs. non-human viral protein classification**

| Condition | Accuracy | Std |
|---|---|---|
| Holdout | **98.0%** | ±0.5% |
| SGTM Attn-Only | 94.6% | ±1.9% |
| SGTM Attn-Only ablated | 88.2% | ±1.7% |
| SGTM Attn+MLP | 85.5% | ±10.3% |
| SGTM Attn+MLP ablated | 80.2% | ±1.3% |

**Task 2: Viral vs. non-viral protein classification**

| Condition | Accuracy | Std |
|---|---|---|
| Holdout | **90.0%** | ±5.4% |
| SGTM Attn-Only | 80.0% | ±12.9% |
| SGTM Attn-Only ablated | 81.3% | ±5.8% |
| SGTM Attn+MLP | 83.7% | ±4.8% |
| SGTM Attn+MLP ablated | 78.8% | ±3.4% |

Two observations stand out:

1. **The holdout model is the best viral classifier** (98.0% on Task 1) despite never seeing human-infecting viral proteins during training. It learned enough from adjacent/ambiguous data and general protein structure to classify viruses with near-perfect accuracy.

2. **Ablation barely affects classification despite destroying perplexity.** The attention-only model drops only 6.4 percentage points on Task 1 after ablation (94.6% → 88.2%), and actually *improves slightly* on Task 2 (80.0% → 81.3%). Representations retain viral knowledge even when the model cannot generate coherent token predictions.

## 5. Discussion

### What We Can Say with Confidence

Four empirical findings emerge clearly from these experiments:

**SGTM + ablation fails at 35M for ESM-2.** Both attention-only and attention+MLP ablation produce catastrophic perplexity increases (10²²–10³⁴). The technique does not achieve its stated goal of selective knowledge removal at this scale.

**The 8M attention-only result does not generalize.** What appeared to be a viable ablation strategy at 8M — zeroing two attention heads with only +0.6% perplexity cost — becomes catastrophic at 35M. This is not a graceful degradation; it is a qualitative failure mode transition.

**Data filtering does not prevent viral classification capability.** The holdout model, which never saw human-infecting viral proteins during training, achieves 98.0% accuracy at classifying them. Protein structure is shared across domains: a model trained on general proteins inherently learns representations useful for viral classification.

**Representations retain viral knowledge through catastrophic ablation.** Even when ablation destroys the model's token prediction mechanism (PPL → 10²²), linear probes show 80–88% classification accuracy on viral proteins. The knowledge lives in the geometry of the representation space, not solely in the parameters that were ablated.

### Competing Hypotheses

These results are consistent with multiple explanations. We flag the following explicitly as hypotheses, not established conclusions:

**Hypothesis 1: Residual stream entanglement scales with depth.** In transformers, every layer reads from and writes to a shared residual stream. With 12 layers (vs. 6 at 8M), there are twice as many opportunities for retain parameters to develop dependencies on forget head outputs. When forget heads are zeroed, the distribution shift compounds through all downstream layers. This is a plausible mechanism for the 8M-to-35M failure transition, but confirming it would require causal tracing experiments that we have not yet conducted.

**Hypothesis 2: The capacity allocation is wrong.** We allocated 2 of 20 attention heads (10%) as forget parameters at 35M. This may be too few or too many. Additionally, the MLP forget allocation (12.5% of intermediate dimensions) may be poorly chosen. A 1-head experiment is currently in progress to test whether tighter capacity allocation changes the outcome.

**Hypothesis 3: Perplexity is an incomplete evaluation.** Perplexity measures the model's token prediction capability, but the linear probe results show that representation-level knowledge and token-level generation are partially decoupled. A model could conceivably have low perplexity on retain data while still encoding dangerous knowledge in its representations — or, as we observe, have destroyed perplexity while retaining representational knowledge. A more targeted evaluation framework (e.g., BioRiskEval) would better capture what "dangerous capability" means for protein models.

**Hypothesis 4: Biological knowledge may be fundamentally non-separable.** The linear probe results — particularly the holdout model's 98% accuracy on viral classification without viral training data — provide a Bayesian update toward the view that viral and non-viral protein knowledge are not cleanly separable. Viral proteins share folding patterns, amino acid distributions, and structural motifs with all other proteins. If the knowledge overlap is in the biology itself rather than in the model's compression scheme, then no weight-level intervention can separate them cleanly. This is a strong claim, and our experiments do not confirm it — but the linear probe results are consistent with it.

### What We Don't Know

Several important questions remain unresolved:

- **Whether different ablation strategies could work.** We tested only zeroing. Noise reinitialization, scaling, or learned ablation masks might produce different outcomes.
- **Whether autoregressive protein models behave differently.** ESM-2 is a masked encoder. Autoregressive protein generators (e.g., Evo2-style models) have a generation step that creates additional intervention surface. SGTM's original success was on autoregressive models, and the failure may be architecture-specific.
- **Whether results at 8M–35M transfer to frontier scale.** The SGTM paper's smallest successful model was GPT-2 Small at 117M — roughly 3× our largest model. There may be a scale threshold below which SGTM cannot work due to insufficient parameter redundancy.
- **Whether the linear probe results reflect true capability or statistical artifacts.** The probes operate on mean-pooled embeddings with small evaluation sets (55 forget, 15 adjacent). While the holdout's 98% accuracy is striking, the high variance on some conditions (±10.3% for SGTM Attn+MLP on Task 1) suggests limited statistical power.

## 6. Implications

These results, while preliminary, suggest several tentative conclusions about weight-level safeguards for protein language models.

**Protein models may face harder safeguard challenges than LLMs.** The absence of a knowing-versus-helping gap means that every intervention must operate on the knowledge itself — there is no behavioral layer to exploit. Combined with the deep biological overlap between dangerous and benign protein knowledge, this creates a fundamentally more constrained intervention surface than text-based models offer.

**Even broken generation leaves exploitable representations.** The linear probe results suggest that destroying a model's ability to predict tokens (via ablation) does not destroy the information encoded in its intermediate representations. An adversary who fine-tunes an ablated model on a small amount of viral data may be able to recover generation capability by rebuilding output pathways on top of intact representations. Any viable approach to knowledge removal must address the representation geometry, not just the output mechanism.

**Data filtering is currently the dominant strategy, but has known limitations.** Our holdout model produces the best general performance (retain PPL 10.23 vs. 13.78–13.92 for SGTM) and is the simplest to implement. However, the 98% linear probe result demonstrates that data filtering does not prevent the model from acquiring the target capability — it simply prevents the model from being *trained on* target data. For domains where structural similarity makes capability acquisition inevitable, data filtering's protection may be more nominal than real.

**The gap between generation capability and representation knowledge motivates exploring deployment-context interventions.** If weight-level interventions cannot reliably remove knowledge from representations, then complementary approaches — access controls, usage monitoring, output screening — become more important, even for open-weight models where they are harder to enforce. This does not mean weight-level interventions are worthless, but it does mean they are unlikely to be sufficient in isolation.

## 7. Future Directions

### Near-Term Experiments

**1-head experiment (in progress).** We are currently training a 35M SGTM variant with a single forget attention head, testing the hypothesis that tighter capacity allocation improves ablation outcomes. If 2 heads is too many — forcing the model to route general computation through forget heads — then 1 head may preserve cleaner separation.

**BioRiskEval benchmark integration.** Our current evaluation relies on perplexity and linear probes, which measure general language modeling and linear separability of representations respectively. Neither directly measures what we care about: whether the model provides *marginal uplift* for a biosecurity-relevant task. BioRiskEval (or a domain-specific variant) would provide a more targeted assessment of whether ablation meaningfully degrades dangerous capabilities versus simply destroying the model.

### Medium-Term Directions

**Autoregressive protein models.** ESM-2 is a masked encoder — the threat model that motivates this work (open-weight release of a model that can generate dangerous proteins) is better captured by autoregressive generators like Evo2. Testing SGTM on an autoregressive protein model would clarify whether our results are specific to the encoder architecture or reflect a deeper challenge. The original SGTM paper's success on autoregressive models (GPT-2) suggests this is worth investigating.

**Post-hoc surgery methods.** SGTM's core practical limitation is that each configuration requires a full training run to validate. Post-hoc methods — task vectors, causal tracing, ROME/MEMIT-style edits, LEACE concept erasure — operate on already-trained checkpoints and can be iterated cheaply (minutes per trial vs. hours per training run). Even if none achieves clean separation, the cost structure allows systematic exploration of the knowledge-capability tradeoff curve.

**Representation engineering.** Rather than removing parameters, representation engineering identifies directions in activation space corresponding to target concepts and suppresses them. This operates in representation space rather than parameter space, potentially degrading more gracefully than the all-or-nothing failure mode of parameter ablation.

### Longer-Term Questions

**Alternative safeguard paradigms.** Our experiments, combined with the broader unlearning literature, suggest that *removing* dangerous knowledge from open-weight models may be fundamentally harder than *never learning* it (data filtering) or *controlling access* to models that have it (deployment context). This motivates exploring safeguard approaches that don't require clean knowledge separation:

- *Entangled safeguards*: Can safety behavior be woven so deeply into a model's computation that removing it requires effectively retraining from scratch? For protein models, the deep biological entanglement between pathogenicity and protein structure may make this more tractable than for text models — pathogenicity *is* a structural property, so a model that understands structure necessarily engages with pathogenicity-relevant features at every layer.
- *Self-destructing models*: Could a model be designed such that fine-tuning on hazardous data triggers catastrophic degradation — not of the target knowledge, but of the general capability that makes the model useful? This inverts the ablation approach: instead of trying to remove dangerous knowledge, you make the model destroy itself if an adversary tries to recover it.

**Compute governance and access controls.** Weight-level safeguards are one piece of a larger puzzle. Even if no weight-level intervention works perfectly, the combination of partial weight-level protection, compute requirements for retraining, and monitoring of access patterns may collectively raise the barrier to misuse above the threshold where the model provides marginal uplift. Understanding where the real leverage lies — and whether it is technical (better safeguards) or structural (who has access to what) — is a strategic question that our technical results inform but do not resolve.

**The protein model threat landscape is evolving.** Frontier protein models (Evo2, ESM3) are substantially larger and more capable than the models tested here. Whether weight-level safeguard challenges intensify or diminish at frontier scale is an empirical question that cannot be answered at 8M–35M. Our results provide a lower bound on the difficulty of the problem; the upper bound remains unknown.

## 8. Summary

We conducted the first empirical test of SGTM on protein language models, training ESM-2 at 8M and 35M parameter scales with gradient masking targeting human-infecting viral protein knowledge. The results are negative: SGTM ablation that appeared viable at 8M fails catastrophically at 35M, and linear probes reveal that both data-filtered and ablated models retain viral classification capabilities in their representations.

These findings contribute to the broader question of whether weight-level safeguards are viable for open-weight protein language models. The answer, as of this midpoint, is "not with current techniques, and possibly not in principle for knowledge that is biologically entangled with general protein understanding." The remainder of this project will test these boundaries — varying capacity allocation, evaluation frameworks, and architectural assumptions — while also exploring whether alternative safeguard paradigms (entangled safeguards, representation engineering, deployment-context interventions) offer more promising paths forward.

---

*Total compute used: ~$90 across 8M and 35M experiments. Infrastructure: RunPod RTX 4090 instances.*
