# Phase 2 Research Log: Post-8M/35M Experiments

**Project:** Weight-Level Safeguards for Open-Weight Protein Language Models
**Author:** Lee
**Started:** 2026-02-27
**Fellowship:** ERA AI x Biosecurity

---

## Phase 1 Summary

Phase 1 (8M and 35M ESM-2 SGTM experiments) produced a clear negative result: SGTM gradient masking, which works on GPT-2 117M–774M in the original paper, fails catastrophically when applied to ESM-2 at 35M scale. Post-ablation perplexity jumps to 10²²–10³⁴ (versus +0.6% at 8M with attention-only masking). See `midpoint_report.md` for full writeup.

Key open questions from Phase 1:
- Is SGTM failure due to masked LM architecture, model scale, or hyperparameter choices?
- Do protein models face fundamentally harder safeguard challenges than LLMs?
- Can data filtering (holdout) be easily reversed by fine-tuning?

---

## Phase 2 Goals

1. **Fix linear probe methodology** — Re-evaluate with held-out val splits, balanced classes, balanced accuracy scoring
2. **Recovery fine-tuning experiment** — Measure how quickly holdout models recover viral capability
3. **Scale to 150M** — Enter the bracket of the original SGTM paper (117M–774M)
4. **Explore autoregressive protein models** — Test whether architecture (encoder vs decoder) matters
5. **Strengthen eval** — Move beyond PPL toward biologically meaningful benchmarks

---

## Experiment Log

### 2026-02-27: Linear Probe Methodology Fix

**Problem identified:** The original `linear_probe.py` had two critical issues:
1. **Data leakage:** Loaded all raw TSV sequences (including training data) for evaluation, not held-out val splits
2. **Class imbalance:** Task 1 had 78% majority-class baseline (1239 vs 349), Task 2 had 76% baseline (1588 vs 500). Post-ablation accuracies of 80–88% were barely above chance.

**Fix applied:**
- `load_virus_sequences()` now loads from `data/sgtm/forget/val` and `data/sgtm/adjacent/val`
- `load_viral_vs_nonviral()` now loads from val splits and balances classes
- Scoring changed from `accuracy` to `balanced_accuracy`

**Status:** Code fixed, awaiting re-run on GPU.

**Impact on claims:** The midpoint report's claim that "representations survive ablation" (based on 80–88% probe accuracy) may weaken substantially once evaluated properly. The probe results need replication before they can support any conclusions.

---

### 2026-02-27: Recovery Fine-Tuning Experiment Design

**Question:** If we train a model with data filtering (holdout — never sees forget-set sequences), how quickly does it recover viral protein capability when an adversary fine-tunes on the filtered data?

**Why this matters:** Data filtering is the current "best available" safeguard for open-weight models. If capability recovers in ~100 steps of fine-tuning, filtering provides negligible security against a motivated adversary with the filtered data.

**Protocol:**
1. Load holdout checkpoint (`models/sgtm/holdout/final_model.pt`)
2. Fine-tune on forget data only (999 human-infecting viral proteins, MLM objective)
3. Evaluate PPL on forget/adjacent/retain every 50 steps
4. Compare recovery curve to original training convergence

**Key metrics:**
- Steps to match baseline forget PPL (how fast does capability return?)
- Retain PPL degradation (does recovery hurt general capability?)
- Adjacent PPL (does non-human viral knowledge transfer?)

**Script:** `sgtm/recovery_finetune.py`

**Status:** Script written, awaiting GPU access.

---

### Upcoming: 150M ESM-2 Experiments

**BOTEC:** ~$88–264 depending on conditions and mixed precision. At minimum need holdout + 1 SGTM condition.

**Decision needed:** Which SGTM configuration to test at 150M? Options:
- Same proportional allocation (last 10% of heads)
- 1-head experiment (minimal capacity, tests whether less is more)
- Multiple configurations (expensive but more informative)

**Status:** Planning. Need to add 150M config to `model_config.py` first.

---

### Upcoming: Autoregressive Protein Model Options

Candidates identified in Phase 1 research:
- **RITA Small (85M):** GPT-style, protein sequences, good training code available
- **ProGen2-Small (151M):** Autoregressive, closest to SGTM paper's architecture class
- **ProFam-1 (251M):** Full training code available, but larger than ideal
- **GPT-Neo path:** Use SGTM repo's own GPT-Neo architecture with a protein tokenizer

**Decision needed:** Whether to pursue this alongside 150M ESM-2, or as a follow-up.

---

### 2026-03-02: Design Review — Deviations from Original SGTM Paper

After cross-checking our implementation against the original SGTM repo (`ERA AIxBio/SGTM/selective-gradient-masking/`), we identified several deviations that may confound our results. This entry documents the issues and design decisions for the next round of experiments.

#### Issue 1: Forget capacity over-allocation

The original SGTM paper allocates ~3.1% of parameters to the forget partition (e.g., 1/32 heads + 16/512 MLP dims for their 8M TinyStories model). Our experiments allocated substantially more:

| Setup | Forget attention | Forget MLP | Approx. forget % |
|---|---|---|---|
| **Original paper (8M)** | 1/32 heads (3.1%) | 16/512 dims (3.1%) | ~3.1% |
| **Our 8M** | 3/20 heads (15%) | 160/1280 dims (12.5%) | ~13% |
| **Our 35M** | 2/20 heads (10%) | 240/1920 dims (12.5%) | ~11% |

The paper explicitly notes that over-allocating forget capacity hurts retain performance. Our pre-ablation retain PPL penalty (3.5–3.7 points vs holdout) may partly reflect this over-allocation rather than a fundamental property of protein models.

**Decision:** Next experiments should use ~3–5% forget allocation to match the paper's validated range. For 35M (20 heads), this means 1 head + a proportionally small MLP slice.

#### Issue 2: Adjacent vs. ambiguous — redundant categories

The original SGTM paper has three data categories: forget, retain, and adjacent (optional, used in their Wikipedia experiments). Our pipeline has four: forget, retain, adjacent, and ambiguous. The distinction between our "adjacent" (279 non-human viral from curated TSVs) and "ambiguous" (~13,800 viral proteins detected by Swiss-Prot header keywords) is an artifact of our data pipeline, not a meaningful conceptual distinction. Both are viral proteins that aren't in the forget set, and both train with `sgtm_mode="default"` (all parameters update).

Having two overlapping categories adds complexity without clear benefit. Worse, routing ~13,800 ambiguous viral proteins through `default` mode means all parameters — including retain parameters — learn from a large volume of viral data. This could undermine the separation SGTM is supposed to create.

**Decision:** Merge adjacent and ambiguous into a single "adjacent" category, matching the original paper's three-way split (forget/adjacent/retain). Need to decide: should the merged adjacent category include ALL Swiss-Prot viral proteins, or just the curated non-human set?

#### Issue 3: Attention-only ablation not in original paper

The original SGTM paper always masks attention AND MLP together. Our "attention-only" condition was our addition, motivated by the 8M result where it appeared viable (+0.6% retain PPL). At 35M it failed identically to attn+MLP.

**Decision:** Drop the attention-only condition from future experiments. Test only the paper's standard configuration (attention + MLP together) with the corrected forget allocation.

#### Issue 4: Forget task granularity

Our current forget task is fine-grained: forget human-infecting viral proteins but retain non-human viral proteins. This is a subtle biological distinction — a human-infecting influenza protein and a bird-infecting influenza protein are structurally very similar.

**Intuition:** For small models with limited capacity, this fine-grained separation may be asking too much. The model may not have enough parameters to develop separate representations for "human-infecting viral" vs "non-human viral" — they may share too much of the same computational pathway. A coarser task — forget ALL viral proteins, retain only non-viral — would be a much cleaner separation target, since viral and non-viral proteins are structurally distinct categories.

This gives us a natural experimental ladder:
1. **Coarse task (viral vs non-viral):** Easiest. If SGTM can't do this, it can't do anything useful on protein models.
2. **Fine task (human-infecting vs other viral):** Harder. Closer to the real biosecurity use case but may require larger models or better techniques.

Testing the coarse task first would tell us whether the SGTM failure is about the technique's interaction with protein models generally, or specifically about the difficulty of the fine-grained task.

**Decision:** Add a coarse-task condition to the next round of experiments. This requires restructuring the data splits: forget = all viral proteins, retain = all non-viral proteins (no adjacent category needed).

#### Issue 5: Data provenance not documented

The virus_human.tsv and virus_nonhuman.tsv files are clearly UniProt/Swiss-Prot exports (UniProt accession IDs, standard column format, `Virus hosts` column with NCBI TaxIDs). The human/non-human split was done by whether the `Virus hosts` field includes `Homo sapiens [TaxID: 9606]`.

However, there is no download script, no query documentation, and no record of the exact selection criteria. The files appeared in the initial git commit as pre-existing artifacts. We need to:
1. Reconstruct the UniProt query that produces these datasets
2. Create a reproducible download script
3. Document the query date and any additional filters

**Update (2026-03-02):** Analysis confirmed the data is from UniProt/Swiss-Prot. The human/non-human split was done at the *virus species* level, not by the `Virus hosts` annotation field. Specifically:

- 964 of 1,239 "human" entries have `Homo sapiens [TaxID: 9606]` in their Virus hosts field
- 275 entries do NOT — these are all Influenza A strains annotated as avian/swine-only (e.g., A/Duck/England/1/1956 H11N6). They were included because Influenza A as a species infects humans, even though these specific strains don't.
- The "human-infecting" file is 77% Influenza A (955/1239), with the rest being HIV, SARS-CoV-2, etc.
- The "non-human" file is 67% bacteriophage T4 (235/349) and 13% plant viruses

This is problematic for the fine-grained forget task — 22% of the "forget" set isn't actually human-infecting by strain. It also means the fine-grained task is essentially "forget Influenza A + HIV + SARS-CoV-2" rather than a principled biological boundary.

**Resolution:** Created `data/download_virus_data.py` for reproducible UniProt downloads. For future experiments, the coarse task (all viral vs non-viral) is the primary experiment. A better curated forget set is needed for any fine-grained experiments.

#### Summary: Next Round Design

| Parameter | Phase 1 | Phase 2 (proposed) |
|---|---|---|
| Forget allocation | 10–15% of heads, 12.5% of MLP | ~3–5% (1 head, small MLP slice) |
| Data categories | 4 (forget/adjacent/ambiguous/retain) | 3 (forget/adjacent/retain) or 2 (forget/retain for coarse task) |
| Forget task | Human-infecting viral vs non-human viral | Both coarse (all viral) AND fine (human-infecting only) |
| Ablation conditions | Attn-only + Attn+MLP | Attn+MLP only (matches paper) |
| Shared biases | Not partitioned | Still not partitioned (note as limitation) |
| Data provenance | Undocumented | Reproducible download script |

---

### 2026-03-02: Cross-Architecture Critique (from BERT-SGTM control experiment)

A parallel BERT-SGTM experiment (using the same TinyStories data as GPT-Neo to isolate architecture from domain) produced a detailed critique of our ESM-2 experimental design. This section documents each observation, what we've done about it, and what remains open.

#### Addressed in code

**Evaluation masking determinism (critique #4):** Pre- and post-ablation PPL must use identical masked positions. We were relying on global `torch.manual_seed(42)` resets, which is fragile — any operation consuming global RNG state between resets silently changes the masks. **Fixed:** `MLMCollator` now accepts a `seed` parameter that creates a dedicated `torch.Generator`, isolating eval masking from all other RNG consumers. `evaluate_sgtm.py` resets the collator's generator before each eval pass.

**Perplexity incomparability (critique #2):** Absolute MLM PPL and autoregressive PPL are fundamentally different metrics. Cross-architecture comparison requires dimensionless ratios. **Fixed:** `evaluate_sgtm.py` now computes and stores **ablation ratios** (PPL_post / PPL_pre per split) in the results JSON alongside absolute PPL. Primary reporting for cross-architecture comparison should use these ratios.

#### Already fine for ESM-2

- **Pre-LayerNorm (critique #3):** ESM-2 natively uses Pre-LN, matching GPT-Neo. No confound.
- **Shared biases unmasked (critique #6):** Known limitation, consistent across both implementations. Already documented above.
- **Gradient masking per micro-batch (critique #7):** Our `adjust_gradients()` runs per micro-batch inside the accumulation loop. This is idempotent (zeroing already-zero gradients), so functionally equivalent to calling once after accumulation. Not a bug.
- **Attention pattern (critique #9):** ESM-2 uses all-global bidirectional. No confound.
- **Data domain (critique #10):** Already addressed by redesigning the forget task (coarse/fine). The BERT-SGTM experiment controls for this by using the same English/Spanish data as GPT-Neo.

#### Open design decisions (require budget/scope discussion)

**Training signal mismatch (critique #1):** MLM predicts ~15% of tokens per step vs 100% for autoregressive. At 40K steps, ESM-2 gets ~6K effective token predictions per position vs 40K for GPT-Neo. A "signal-matched" condition would need ~267K steps (6.7× compute). This is a real confound for cross-architecture interpretation: if SGTM fails on ESM-2, it could be undertrained rather than architecturally incompatible. **Decision deferred:** Expensive (6.7× per condition). Consider running one signal-matched condition at 35M if budget allows, or noting as a limitation. The BERT-SGTM experiment runs both step-matched and signal-matched.

**Single seed (critique #5):** All ESM-2 experiments use seed=42 with no variance reporting. SGTM involves stochastic data ordering and weight initialization, both of which affect localization quality. Running 3 seeds for the SGTM condition would 3× the SGTM training budget. **Decision deferred:** Note as limitation. If a condition shows promising results, rerun with 2 additional seeds before claiming success.

**Weight decay discrepancy (critique #8):** We use `weight_decay=0.01` (ESM-2 convention) vs GPT-Neo's `0.1`. Higher weight decay encourages sparser representations, potentially making SGTM's localization task easier. This is a confound for any ESM-2 vs GPT-Neo comparison. **Decision deferred:** The BERT-SGTM experiment uses 0.1 to match GPT-Neo. For our ESM-2 experiments, matching ESM-2's original training convention is defensible. Could add a `weight_decay=0.1` condition if budget allows. Note as limitation.

**TODO:** Revisit these open decisions when updating the methods section.

---

## Data Reference

### Phase 1 (8M/35M experiments — superseded)

| Split | Description | Train | Val | Test |
|-------|-------------|-------|-----|------|
| Forget | Human-infecting viral proteins | 999 | 55 | 57 |
| Adjacent | Non-human viral proteins | 279 | 15 | 17 |
| Ambiguous | Swiss-Prot viral + 10% of virus sets | ~13,800 | — | — |
| Retain | Filtered Swiss-Prot non-viral | ~403K | ~10K | ~10K |

### Phase 2 (new pipeline with `--forget-task` flag)

**Coarse task** (`--forget-task coarse`): forget = ALL viral, retain = non-viral. No adjacent.

**Fine task** (`--forget-task fine`): forget = human-infecting, adjacent = other viral (merged from old adjacent + ambiguous), retain = non-viral. Needs a better curated forget set.

**Custom task** (`--forget-task custom --forget-tsv PATH`): forget = user-specified, adjacent = remaining viral, retain = non-viral.

Data generated into `data/sgtm/{coarse,fine,custom}/` subdirectories.

---

## Infrastructure

- **Local:** MacBook (MPS) — 8M smoke tests only, MPS produces worse embeddings than CUDA
- **GPU:** RunPod RTX 4090 24GB or Lambda instance (ubuntu@192.222.51.125)
- **Wandb project:** `sgtm-esm2-8m`, `sgtm-esm2-35m` (cluttered — need to identify meaningful runs)
- **Cost so far:** ~$45 for Phase 1 (three 35M conditions, 28 hrs each)
