# ESM-2 35M SGTM Experiment Options

## Option A: Core comparison (~$40-45) [SELECTED]

3 pods:
1. **Holdout** — all data minus forget set (data filtering control)
2. **SGTM 2head+MLP zeroing** — tests if 35M scale rescues MLP ablation (catastrophic at 8M)
3. **SGTM 2head attention-only zeroing** — tests attention-only at 35M scale

Key question: Does the 35M model's extra capacity (12 layers, 480-dim) let it survive MLP ablation that was catastrophic at 8M?

## Option B: Add 1-head variant (+~$15)

4. **SGTM 1head+MLP zeroing** — fewer forget heads, tests if tighter capacity helps

Decision point: Only if Option A shows MLP ablation is still catastrophic at 35M.

## Option C: Add gradient routing (+~$15)

5. **SGTM 2head routing** — gradient routing instead of zeroing

Decision point: Only if zeroing results are ambiguous or if we want to compare strategies.

## 35M Architecture

- 12 layers, 20 attention heads, head_dim=24, embed_dim=480, MLP=1920
- Forget heads: [18, 19] (2 heads = 10% of attention capacity)
- Forget MLP start: 1680 (240 forget dims = 12.5% of 1920)

## Changes from 8M experiments

- Viral Swiss-Prot filtering: 8M experiments did NOT filter Swiss-Prot by viral taxonomy headers. ~16K viral proteins leaked into the retain set. Fixed for 35M via FASTA header keyword matching.
- head_dim: Parameterized (was hardcoded to 16). 35M uses head_dim=24.
- forget_mlp_start: Config-driven (was hardcoded to 1120). 35M uses 1680.
- Seed reset before ablation eval to ensure identical MLM masks pre/post.
- Linear probe evaluation added for downstream task measurement.
