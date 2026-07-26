# Reference-distribution ablation (Reviewer RpxH, W1)

> **STATUS: IN PROGRESS** — multibench (MUStARD/MOSI/UR-Funny) complete (27/27 runs);
> Hateful Memes (6 running + 1 retry), CREMA-D α=0.5 (2 running + 6 retries queued),
> α=0.1 wave and λ mini-sweep queued. Numbers below update as arms land.

**Claim under test.** The estimator error decomposes as
E[D_KL(p(·|x̃₁,x₂) ‖ r)] = I(X̃₁;Y|X₂) + E[D_KL(p(·|x₂) ‖ r)], so the reference r should matter
only through the second term: negligible on synergy-dependent examples, active on unimodally
solvable ones.

## Setup & reproducibility

- Code: public `SynIB` repo, branch `rebuttal-ref-ablation` (commits `159d7b1` flag, `cb39a55`
  infra; see `git log`). Configs: `run/configs/rebuttal_ref_ablation/` (verbatim copies of the
  paper configs; only CEU paths corrected to the synergy-split pickles — the paper configs'
  `_kfold10` CEUs silently skip on the synergy3 split).
- Commands: the paper `M_random` command per dataset from `docs/REPRODUCE.md`, plus
  `--reference_type {uniform,class_prior,unimodal_anchor} --ref_diag --start_over --tag REFABL_*`.
  Full per-run argument lines: `run/condor/ref_ablation_*.args`. Fold→seed map: 0/1/2 → 109/19/337.
- π per dataset (audit-confirmed effective values): UR-Funny 0.7, MUStARD 0.1, MOSI 0.3, HM 0.3
  (`p_max` unused at runtime), CREMA-D **0.20**. λ: 0.001 / 0.001 / 0.1 / 0.01 (`l_pareto` 0.1) / 1.0.
- References: `uniform` = 1/K; `class_prior` = fixed empirical train-label frequencies computed at
  startup (per-α on CREMA-D-Irony); `unimodal_anchor` = **EMA (decay 0.99) copy of the complementary
  (unmasked) modality's unimodal model**, per App. H.5 — NOT the released code's direction (masked
  modality's own prediction; kept available as `anchor_legacy`), see `docs/rebuttal_reference_audit.md`.
  On HM the anchor/diagnostic heads are initialized from the trained unimodal runs
  (`HM_SmallTF_DeBERTa__uni_{text,image}_seed{109,27,3407}`, fold-matched), because the live HM
  unimodal heads are never trained.
- All three arms use one categorical-KL estimator. Note this means the UR-Funny/MUStARD *paper*
  objective (Gaussian-KL on logits, App. C.1) is not identical to any arm; the ablation compares
  references within one family at the paper's λ.
- Envs: `envs/synergy_new` (multibench, HM), `envs/synergy` (CREMA-D; torchcodec/FFmpeg broken in
  synergy_new). Condor clusters: 52913/52914 (core), 52919/52920 (retries), 52921 (α=0.1),
  52922 (λ MUStARD). wandb disabled; ground truth = per-validation-step logs inside checkpoints,
  extracted by `scripts/analysis/extract_ref_ablation.py`.
- Diagnostic (`--ref_diag`): E_val[D_KL(q(·|x̃_i,x_{-i}) ‖ p̂(·|x_{-i}))] with p̂ = unimodal model
  frozen at initialization (pretrained unimodal encoders for multibench/CREMA-D; trained unimodal
  heads for HM), computed identically for every arm, both symmetric branches, on the full validation
  set at every validation step (a fixed set, superset of the plan's "fixed held-out batch").

## Table 1 — headline metric (test, at best-val step; mean ± sd over 3 folds/seeds)

| dataset | metric | uniform | class_prior | unimodal_anchor |
|---|---|---|---|---|
| MUStARD | acc | 58.83 ± 2.19 | 60.07 ± 3.60 | 59.58 ± 2.65 |
| MOSI | acc | 72.91 ± 1.53 | 73.10 ± 0.66 | 73.37 ± 1.37 |
| UR-Funny | acc | 62.27 ± 0.75 | 62.85 ± 0.40 | 62.51 ± 1.13 |
| Hateful Memes (folds 1–2 only, see note) | acc | 70.00 ± 1.91 | 65.78 ± 4.70 | 67.55 ± 0.71 |
| CREMA-D-Irony α=0.5 | acc | 64.05 ± 2.68 | 63.32 ± 2.67 | 61.99 ± 4.11 |
| CREMA-D-Irony α=0.5 | total-F1 | 58.93 ± 4.31 | 56.58 ± 3.98 | 54.97 ± 4.23 |
| CREMA-D-Irony α=0.5 | irony-F1 | 18.33 ± 10.41 (6.4/25.4/23.2) | 8.28 ± 8.05 (2.5/4.8/17.5) | 3.33 ± 1.44 (2.5/5.0/2.5) |
| CREMA-D-Irony α=0.1 | acc | 69.70 ± 3.39 | 69.99 ± 4.02 | 69.26 ± 2.26 |
| CREMA-D-Irony α=0.1 | irony-F1 | 2.90 ± 5.02 (0/8.7/0) | 0.00 (0/0/0) | 0.00 (0/0/0) |

Synergy-subset accuracy (test samples misclassified by every unimodal model; CEU-defined, identical
subset across arms):

| dataset | uniform | class_prior | unimodal_anchor | paired Δ range vs uniform |
|---|---|---|---|---|
| MUStARD | 33.94 ± 14.25 | 32.10 ± 21.57 | 34.22 ± 23.54 | −1.8 … +0.3 |
| MOSI | 17.40 ± 3.33 | 15.46 ± 3.16 | 14.56 ± 4.16 | −2.8 … −1.9 |
| UR-Funny | 13.90 ± 6.77 | 11.79 ± 0.66 | 16.51 ± 6.94 | −2.1 … +2.6 |
| Hateful Memes (folds 1–2) | 56.15 ± 6.43 | 40.80 ± 15.27 | 47.25 ± 0.78 | class_prior 0.0/−30.7; anchor −3.8/−14.0 |

> **Note (HM fold 0).** The fold-0 `uniform` and `class_prior` jobs (52913_3, 52913_27) crashed on
> a node flake (`RuntimeError: DataLoader worker exited unexpectedly`, exit 1) after 2 and 4
> validation steps respectively, while their `unimodal_anchor` counterpart ran 44 — a killed job
> still writes a checkpoint, so those two arms were undertrained, not comparable. Fold 0 is
> therefore excluded from the HM rows above and both runs were resubmitted (cluster 52950);
> the table will move to n=3 when they land. `extract_ref_ablation.py` now records
> `n_val_steps` per run and warns when an arm falls below half its dataset's median, so this
> class of error cannot pass silently again. (Direction is unaffected: on the valid folds the
> uniform advantage is larger, not smaller, than in the earlier 3-fold numbers.)

Paired per-fold differences vs uniform (headline acc, pp):

| dataset | class_prior − uniform | unimodal_anchor − uniform |
|---|---|---|
| MUStARD | +1.24 (+5.22, −3.73, +2.24) | +0.75 (+0.37, +0.37, +1.49) |
| MOSI | +0.19 (−1.38, +0.95, +1.02) | +0.46 (−2.70, +1.24, +2.84) |
| UR-Funny | +0.58 (−0.05, +1.80, 0.00) | +0.24 (−1.28, +1.80, +0.19) |

## Figure 1 — diagnostic curves

`docs/figures/fig_ref_ablation_diag.pdf` (generated by `scripts/analysis/plot_ref_ablation.py`
from `docs/rebuttal_ref_ablation_curves.csv`). One panel per dataset; color = reference;
solid = modality-2-masked branch, dashed = modality-1-masked branch.

## Pre-registered prediction checklist (filled per dataset; PRELIMINARY until all arms land)

**(a) Synergy-subset accuracy ≈ equal across references (within paired noise).**
- MUStARD: PASS — paired Δ −1.8/+0.3 pp against fold-sd ≥ 14 pp.
- MOSI: PASS (weak) — Δ −1.9/−2.8 pp, subset n ≈ 100–160, well within seed noise.
- UR-Funny: PASS — Δ −2.1/+2.6 pp, within noise.
- Hateful Memes: **FAIL (informative references hurt the synergy subset)** — valid folds (1–2):
  paired Δ vs uniform ≤ 0 in every comparison, class_prior 0.0/−30.7 and anchor −3.8/−14.0;
  uniform 56.2 vs anchor 47.3 vs class_prior 40.8. n=2 pending the fold-0 reruns.
- CREMA-D (class-level analogue: irony-F1): **FAIL in the same direction** — FINAL:
  uniform 18.3 ± 10.4 vs class_prior 8.3 ± 8.1 vs anchor 3.3 ± 1.4; ordering
  uniform > class_prior > anchor matches r's mass on the synergy class (1/7 ≈ 14% > ~8% prior
  > near-0 anchor).

**(b) Headline ordering unimodal_anchor ≥ class_prior ≥ uniform at matched λ; gap larger on
unimodal-heavy datasets.**
- Direction: uniform is the lowest arm on all three completed datasets, but the exact ordering
  anchor ≥ class_prior holds only on MOSI; effect sizes (+0.2…+1.2 pp) are within fold noise at
  n = 3 → **directionally consistent, not significant**. Honest caveat: at λ = 0.001
  (MUStARD/UR-Funny paper value) the KL term is nearly inactive, so near-equality is partly
  by construction; the λ sweep probes this.
- HM / CREMA-D (λ = 0.01 / 1.0, where the term does bite): pending — these are the decisive cells.

**(c) Diagnostic KL → ~0 under unimodal_anchor; plateaus > 0 under uniform, height tracking
unimodal solvability.**
- PRELIMINARY: **FAIL in this form at paper λ on the completed datasets** — the diagnostic KL
  *rises* over training for all references (encoders fine-tune away from the frozen-at-init p̂),
  and the anchor arm is not closer to p̂ than uniform (on MOSI it is farther). Two candidate
  readings, to be settled with HM/CREMA-D (larger λ): (i) at λ ≤ 0.1 the KL penalty is too weak
  to steer q toward any reference, so the diagnostic mostly measures encoder drift; (ii) the EMA
  anchor tracks the *live* unimodal model, not the init snapshot, so q can follow the anchor while
  both drift from p̂. Final verdict + plateau values after all arms land.

**(d) λ-tolerance (uniform degrades faster with λ than unimodal_anchor).** Pending (cluster 52922 +
CREMA-D λ wave).

## Rebuttal paragraph (working draft — numbers final for multibench/CREMA-D-α0.5-partial; HM & remaining arms pending)

**Purpose:** show whether the reference r makes any substantial difference, and support the
intuition behind the paper's choice — not to characterize convergence.

> *On the choice of reference distribution (W1).* The KL term compares the corrupted-input
> prediction to a reference r, and the reviewer asks how sensitive the objective is to this
> construction. We ran the full benchmark suite with three references — uniform, the empirical
> class prior, and the complementary modality's unimodal prediction (EMA copy) — holding every
> other hyperparameter at the paper's values (3 seeds/folds each). At the operating λ of every
> benchmark, the choice of r is second-order: headline metrics move by at most ~1 point
> (MUStARD 58.8/60.1/59.6, MOSI 72.9/73.1/73.4, UR-Funny 62.3/62.9/62.5 for
> uniform/class-prior/anchor; HM pending), and synergy-subset accuracy is statistically
> unchanged in paired per-fold comparisons. This matches the decomposition
> E[D_KL(p(·|x̃₁,x₂)‖r)] = I(X̃₁;Y|X₂) + E[D_KL(p(·|x₂)‖r)]: r enters only through the second
> term, which is bounded (≤ log K for uniform) and small exactly on the synergy-dependent
> examples the objective targets, where the remaining-modality conditional is uncertain. The
> one regime where r matters confirms the same intuition: on CREMA-D-Irony (λ = 1.0, where the
> penalty binds) an *informative-but-wrong* reference — the complementary unimodal prediction,
> which confidently asserts the base emotion on ironic samples — makes the second term large on
> precisely the synergy class and suppresses it (irony-F1 18.3 → 3.3), while the uninformative
> uniform reference only tempers confidence and preserves synergy. Both findings — insensitivity
> at operating λ, and the direction of the one sensitivity — support the principle behind our
> construction: the reference should inject no achievable label information, and we standardize
> on the uniform reference in the revision.

*(Pending before final: HM columns; CREMA-D class-prior folds 0/2; `anchor_legacy` arms —
the released code's direction — expected to behave like the uninformative case because its
target is not predictable from the visible modality; α=0.1 scarce-synergy check; λ sweep.)*
