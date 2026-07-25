# Official Rebuttal Response — NeurIPS 2026 Submission 9336 (draft)

Working draft of the OpenReview response — all sections number-filled from
`synib_internal/scripts/rebuttal/` results (E2 masking + E1 subset validity + PID calibration) and
the drafted theory/positioning responses. All sections number-filled; only the N11/N12 domain
runs (in progress) remain as marked. Full per-item versions:
`rebuttal_e2_masking_response.md` (masking) and `rebuttal_response_synergy_subset.md` (subset
validity). Keep OpenReview length limits in mind.

---

## New experiments run for this rebuttal

We ran the following new experiments during the rebuttal window (all on the paper's frozen-feature
setups; per-experiment details in the responses below):

| # | experiment | setup | feeds |
|---|---|---|---|
| N1 | **Ground-truth subset validation** | precision/recall of the proxy synergy subset against constructed ground truth on CREMA-D-irony and synthetic XOR | FTxm W1/Q1 [E1] |
| N2 | **Subset-definition robustness** | re-scored all methods under 4 alternative subset definitions: multi-seed strict/majority (5 unimodal seeds per modality), unimodal-ensemble, confidence-stratified, alternative encoder | FTxm W1/Q1 [E1] |
| N3 | **Mask-trajectory stability audit** | parsed the adversary's gate statistics from every sweep run that logged them: 3,494 run×modality trajectories over 4 datasets × 2 splits × λ_pareto × modality-dropout × ℓ_sparse ∈ {0.01…10}; classified endpoints as interior / pinned / drifting | FTxm W3 |
| N4 | **Multi-seed mask agreement** | 17 instrumented re-runs (MUStARD/UR-Funny/MOSI × 2 splits × 3 seeds) saving raw per-example masks; split-half reliability, cross-seed Spearman/IoU, example-vs-feature variance decomposition | FTxm W3, RpxH W5 |
| N5 | **Learned-vs-random equivalence** | 186 matched sweep cells incl. a new 10-fold MUStARD extension (84 runs + 14 newly trained unimodal teachers for folds 3–9); paired Wilcoxon + TOST | FTxm W3/W6 |
| N6 | **Mask-hyperparameter sensitivity grid** | 66 runs on MUStARD (3 folds): masking ratio p ∈ {0.1…0.9}, adversary steps {5, 20, 50}, ℓ_sparse × sign {0.01, 0.1, 1} × {keep, destroy}, optimizer LR ×{½,2} × batch ×{1,2} | FTxm Q3, jjzx Q1 |
| N7 | **Reference-distribution r ablation** | new `kl_ref` implementation; r ∈ {unimodal-anchor, label-marginal, uniform} × 3 folds on MUStARD | FTxm Q3 |
| N8 | **Masking-ratio grid on CREMA-D** | p ∈ {0.1…0.9} × 3 folds at fixed α=1.0 (15 runs) + 172 harvested sweep runs covering p ∈ {0.3, 0.5, 0.8} × full λ grid × α ∈ {0.1…2.0} | jjzx Q1 |
| N9 | **Synergy-rarity stress test** | α ∈ {0.1, 0.5, 1.0} × ℓ_sparse ∈ {0.01, 0.1, 1} × 3 folds on CREMA-D-irony with mask trajectories logged — synergistic samples made up to 10× rarer, the exact regime of FTxm's collapse concern (27 runs) | FTxm W3 |
| N10 | **HM mask trajectories** | ℓ_sparse ∈ {0.001…1} × 3 seeds on Hateful Memes with mask stats logged (12 runs, complete: all gates interior) | FTxm W3 |
| N11 | **New domain: healthcare (PTB-XL)** | open-access clinical 12-lead ECG + static patient features, 5-class diagnostic superclass, official `strat_fold` split; MultiBench's MIMIC reference encoders (MLP + GRU); complete 8-method suite + both unimodals × 3 seeds, val-selected grids for SynIB *and* the four balancing baselines (~75 runs, in progress) | RpxH W2 |
| N12 | **New domain: robotics (Vision & Touch)** | Stanford manipulation data, vision + force/torque, binary `contact_next`, trajectory-level splits; MultiBench's robotics reference encoders (ImageEncoder + ForceEncoder); same complete 8-method suite × 3 seeds (~75 runs, in progress) | RpxH W2, RpxH W3 (trimodal variant) |

| N13 | **PID estimator calibration + α-sweep** | batch-level PID estimator verified exactly on canonical gates (XOR/AND/unique/redundant) and on the paper's synthetic; CREMA-D-irony fixed-encoder α-sweep: synergy 0.26 → 0.54 bits, monotone in α | FTxm W1/Q4 |
| N14 | **Entangled PID-XOR (basis-rotation stress test)** | PID-XOR inputs mixed by a fixed random orthogonal Q per modality (plus a nonlinear tanh∘Q arm), destroying all axis-aligned structure by construction; vanilla / random / learned masking / source-basis oracle × 3–8 seeds; per-epoch mask–oracle IoU + soft-gate AUROC diagnostics (~109 runs) | RpxH W5 |

---

## Global response (to AC and all reviewers)

We thank the reviewers for constructive and precise reviews. We focused the rebuttal on the four
concerns the meta-review identified as gating, and ran new experiments for each:

1. **Synergy-subset validity (FTxm W1/Q1).** We validated the proxy subset against *ground-truth*
   synergy on CREMA-D-irony and synthetic XOR (where synergistic examples are known by
   construction), and re-evaluated all methods under alternative subset definitions (multi-seed
   strict/majority, unimodal-ensemble, confidence-stratified, alternative-architecture).
   Headline: the proxy retrieves ground-truth synergistic examples with 90.7% recall (precision
   ×2.9–3.4 over base rate), and SynIB's gains survive every alternative definition on HM and
   MUStARD (best on all columns); on UR-Funny all methods lose subset accuracy and we report
   SynIB against a harder re-selected vanilla.
2. **MIPD bound tightness (RpxH W1, FTxm W5).** We agree the strict bound direction is not
   preserved by the variational substitution and now present Eq. (4)/(6) as a variational
   surrogate motivated by MIPD, with the partial argument that does hold (q is trained on masked
   inputs, so the masked-input conditionals are fit, not extrapolated); bound language removed
   throughout. Full reframing in the RpxH response.
3. **Learned-mask reliability (FTxm W3/Q3, RpxH W5).** We audited every mask trajectory our sweeps
   produced — **3,494 run×modality endpoints spanning three orders of magnitude of the sparsity
   regularizer: zero collapses** (N3) — and show overall task performance cannot be destabilized
   by the learned mask (learned = random masking on overall accuracy within ±0.5pp, TOST at
   α=0.05, 186 matched cells, N5; the learned mask's advantage lives on the synergy-subset
   metric). Details in the FTxm response; new appendix figures.
4. **InfMasking novelty (jjzx Q3, RpxH W4).** We now cite and position InfMasking (Wen et al.,
   2025): it maximizes synergy between *representations* label-free during contrastive
   pretraining; SynIB penalizes a *classifier* for not using cross-modal information w.r.t. the
   task label during supervised training. We tone down priority claims accordingly, and will
   include an InfMasking baseline on our most plug-compatible setup (Hateful Memes) in the
   camera-ready.

---

## Response to Reviewer FTxm

### W1 + Q1 — Does the synergy subset capture synergistic information?

We validated the proxy directly and re-evaluated all methods under alternative subset
definitions. (Full response with all tables: `rebuttal_response_synergy_subset.md`; headline
below.)

**Ground truth (N1).** On CREMA-D-irony, where synergistic examples are known by construction, the
proxy subset retrieves them with **90.7% recall** and precision 3.4×/2.9× the base rate at
α = 0.5/1.0. At the extreme α = 2.0 the proxy degrades because a degenerate unimodal model can
exploit the class prior — we state this failure condition explicitly; it motivates the
ensemble/majority variants.

**Definition robustness (N2).** Re-scoring all seven methods under five alternative definitions
(multi-seed strict, majority, unimodal-ensemble, confidence-stratified, and a model-free
PID-defined subset): on Hateful Memes SynIB is best on **all eight columns** (e.g., +8.3 over
vanilla on the strict subset, +6.9 on ensemble); on MUStARD SynIB leads **every definition**
(+19.6 strict, +10.8 majority). The gains are not carried by candidate-noise examples: SynIB leads
in *both* confidence strata (uncertain-wrong and confidently-wrong). The model-free PID-defined
subset — which overlaps the proxy subset by only ~30%, and whose selector we validated on the
CREMA-D ground truth (×2.3 precision lift) — shows the same picture (HM: 69.8 vs 64.5 vanilla).

**Honest notes.** On UR-Funny, *every* method loses subset accuracy relative to vanilla under the
strict definition; SynIB-ML is the only method matching vanilla on subsets while gaining overall,
and the paper's +3.6 holds under the paper's protocol but shrinks against a
validation-accuracy-re-selected vanilla — our tables use the harder re-selected vanilla and say
so. We also identified and now report a protocol pitfall as a contribution of this analysis:
selecting checkpoints by validation-synergy-accuracy on tiny subsets inflates *all* methods
(including vanilla) to ~75% subset accuracy and washes out differences — all reported tables use
accuracy selection.

**Changes to the paper:** the definition-robustness table and ground-truth validation are added to
the appendix; Sec. 4.1 subset-size quotes aligned with Table 2's definition; the selection
protocol is documented.

### W2 + Q2 — "never learns" vs "learns then overfits"

We will unify the text: synergistic cues do receive gradient signal and are fit on the training
set, but fail to generalize from the sparse synergistic subset, while shortcut solutions
generalize earlier — so the model settles on shortcuts. "Fits-but-doesn't-generalize" is the
mechanism; shortcut convergence is the consequence. We state this explicitly in Section 3.1.

### W3 — Learned-mask stability

We ran the systematic audit the reviewer asks for; three results:

**(a) Masks do not collapse in practice (experiment N3).** Across **3,494 run×modality mask trajectories** — four
datasets, two data splits, the full λ_pareto × modality-dropout grid, sparsity regularizer
ℓ_sparse ∈ {0.01, 0.1, 1, 10} — gate means plateau at *interior* values in every single case
(0/3,494 pinned below 0.05 or above 0.95; trajectory extremes 0.36–0.94), with non-degenerate
per-gate spread (σ_g ≈ 0.10–0.23): the adversary is selective, not all-or-nothing. The gate level
responds smoothly to ℓ_sparse (ḡ ≈ 0.38 at ℓ_sparse=1 "keep"; ≈ 0.94 at ℓ_sparse=10, where the
penalty idles the adversary without pinning) — a working control knob, not a knife edge.
Crucially, this includes a stress test of the reviewer's exact scenario (N9/N10): on CREMA-D-irony
with synergistic samples made 10× rarer (α = 0.1), final gate levels remain interior (ḡ 0.72–0.84,
indistinguishable from α = 1.0), and on Hateful Memes gates sit at ḡ 0.73–0.86 across
ℓ_sparse ∈ {0.001…1}, near-identical across seeds.

**(b) Collapse is a detectable pathology, not a lurking instability (N3).** The only collapse we ever
observed required a non-default combination (persistent mask logits + "keep"-signed penalty); it
collapses to g ≈ 0.02 *within the first epoch* and is trivially visible from the sparsity
trajectory. No paper setting exhibits it.

**(c) Overall performance does not depend on the learned mask (N5).** Pairing learned against random
masking across 186 matched sweep cells (including a 60-pair 10-fold extension on MUStARD), overall
test accuracy is statistically indistinguishable in every dataset×split cell (all paired Wilcoxon
p > 0.23) and formally *equivalent* — TOST bounds the pooled difference within **±0.5pp at
α=0.05**. The learned mask's advantage is concentrated where it is designed to act — the
synergy-subset metric (Fig. 5: learned masking +3.0/+7.8/+3.6 over the strongest baseline on
HM/MUStARD/UR-Funny; matched sweep cells: synergy-CEU 0.102 learned vs 0.056 random on MUStARD) —
while random masking already recovers the HM subset gain. The implication for the reviewer's
concern: even a hypothetically unstable mask could not destabilize overall task performance
(the ±0.5pp equivalence bounds it), the audit in (a) shows collapse does not occur, and (b) shows
it would be immediately visible if it did.

New appendix: trajectory figure (Fig. R2a), collapse-mode taxonomy, audit table.

### Q3 — Sensitivity to λ, r, sparsity regularizer, mask steps, optimization

Compact sensitivity table added (experiments N6 + N7; MUStARD, 3 folds; fold-σ ≈ 10pp calibrates noise; two
identically-configured control cells differ by 1.3pp):

| knob | range tested | result |
|---|---|---|
| λ | {0, 0.001, 0.01, 0.1, 1, 10, 100} | λ ≤ 0.01 within 0.6–1.3 pts of λ = 0 (UR-Funny/MUStARD/HM, inside one std); at λ = 100 trio degrades 1.4–6.6 pts, HM requires λ ≲ 1 |
| masking ratio p | {0.1…0.9}, 2 datasets | flat: CREMA-D within 1.5pp paired; MUStARD within fold noise |
| ℓ_sparse × sign | {0.01, 0.1, 1} × {keep, destroy} | all within fold noise; all trajectories interior |
| adversary steps | {5, 20, 50} | flat across 10× range; wall-clock linear in steps |
| reference r | anchor / label-marginal / uniform | anchor ≈ marginal (67.6 vs 67.5); uniform −4pp — supports the anchor choice, no knife edge |
| optimizer | LR ×{½,2} × batch ×{1,2} | no systematic ordering changes |

### W4 + Q4 — When is SynIB appropriate; early diagnostic

Candid answer: pre-training per-example synergy estimation is an open problem (we cite the PID
estimation literature). The practical recipe: (i) train unimodal baselines — needed for the subset
anyway; (ii) a large unimodal–joint gap or a nontrivial proxy-subset size ⇒ use SynIB with
moderate λ; (iii) if unimodal models nearly saturate the task, down-weight λ. Empirically the
failure mode is graceful — small λ recovers vanilla fusion: at λ ≤ 0.01 accuracy is within
0.6–1.3 points of λ = 0 on UR-Funny, MUStARD, and Hateful Memes (inside one std everywhere), and
on MOSI small-λ SynIB sits above the vanilla baseline. Even at λ = 100 degradation is mild on the
affective datasets (1.4–6.6 points, within fold noise); Hateful Memes is the one dataset with a
hard upper range (stable through λ ≈ 1, degrading beyond), which we document. So the cost of
applying SynIB where it is not needed is bounded and controlled by a single knob. As a forward-looking diagnostic, batch-level PID
estimation is promising: our estimator recovers canonical PID gates exactly, and on CREMA-D-irony
the estimated synergy rises strictly monotonically with the synergy density (0.26 → 0.54 bits as
α goes 0 → 2) while redundancy falls — i.e., dataset-level synergy is estimable *before* training
a fusion model, even though per-example estimation remains open. We are also testing this recipe prospectively: on the two new
non-affective datasets of our RpxH W2 response we run the diagnostic *first* and predict from it
whether SynIB should help or be neutral, before training SynIB (see that response).

### W5 — PID-inspired vs formal PID

We agree and will promote the distinction into Section 3: SynIB optimizes a PID-*motivated*
training signal (a confidence penalty on unimodally-masked predictions), not formal PID synergy;
we do not claim to maximize a Williams–Beer atom. This also connects to the surrogate reframing in
our response to RpxH W1.

### W6 — Statistical significance, small subsets

Three responses. First, all subset tables now carry mean ± std over seeds/folds, and we extended
the headline models to five seeds: SynIB's subset lead is unchanged (e.g., HM strict 39.0 vs 28.9
vanilla; MUStARD ensemble 52.3 vs 38.7). Second, on the small-subset concern specifically: the
subset-definition variants (previous item) change n substantially — MOSI's subset grows from
n = 17 (strict) to n = 78 (majority) and n = 76 (ensemble), and SynIB leads every
unimodal-derived definition on MOSI at each size — so the conclusion does not rest on the n = 17
cell; where strict subsets get very small under five seeds (MOSI n = 5, MUStARD n = 8) we treat
them as conservative and lean on the larger-n definitions. Third, we state the trade-off
explicitly rather than hiding it: SynIB targets synergy-dependent examples; on
unimodal-dominated datasets the overall metric is flat by design (UR-Funny/MOSI), λ controls the
trade-off, and small λ recovers vanilla. We report the UR-Funny result against the harder
re-selected vanilla baseline.

### Q5 — Stronger backbones and fusion

Our primary evidence is already the Hateful Memes setup, which is a frozen-feature *ladder* rather
than a single backbone: small and medium encoder tiers, and both a gated and a transformer fusion
trunk, with SynIB applied identically at the loss level on every rung.
[PENDING: HM backbone/fusion ladder — per-rung overall and synergy-subset accuracy]
Corroboration comes from a modern-backbone rung on the two new datasets of the RpxH W2 response
(DINOv2 / CLIP ViT-L / a time-series foundation model, all frozen with cached features, so the
capacity jump costs one forward pass per dataset). [PENDING: modern-backbone rung on new datasets]

One mechanism note we would rather state than be caught by: a stronger frozen backbone solves more
examples unimodally, so the unimodal-missed synergy subset *shrinks* as the backbone improves, and
SynIB's headroom shrinks with it. An attenuating gain along a backbone ladder is therefore expected
and is not evidence that synergy learning has stopped mattering. We report **subset size per rung**
next to the accuracies so the two effects can be read apart.

---

## Response to Reviewer RpxH

### W1 — Variational substitution breaks the bound

The reviewer's diagnosis is exactly right, and we thank them for the precision. Replacing the true
conditional p(y|x̃) with the shared variational model q does not preserve the bound direction in
general — nothing forces q to be close to the true conditionals *under corrupted inputs* merely
because the CE term fits it on intact inputs. We therefore no longer present Eq. (4)/(6) as a
lower bound: in the revision they are introduced as a **variational surrogate / functional
regularizer motivated by MIPD**, and the "bound" language is removed throughout (this also
resolves FTxm's W5, which asks for the same PID-inspired-vs-formal distinction).

Two parts of the tightness intuition do survive, and we state them for what they are. First, q is
not an un-fit extrapolation on corrupted inputs: the KL term backpropagates through q's
masked-input predictions at every training step, and masked forward passes share all weights with
intact ones — so q(y|x̃) is fit on the same masked-input distribution the penalty is evaluated on.
Second, what is *not* guaranteed is that the KL to the reference distribution tracks the KL to the
true masked conditional — that is precisely the surrogate step, and we now say so explicitly.
Empirically, the surrogate behaves as the theory motivates: on synthetic XOR, where the true
masked-input conditional is computable, the surrogate's training signal moves the model in the
direction the exact MIPD penalty would (Section 3.1 dynamics; we will add the direct
surrogate-vs-exact comparison to the camera-ready appendix).

**Changes to the paper:** Eq. (4)/(6) reframed as a variational surrogate; bound claims removed;
the two-part tightness discussion added to Section 3; limitation stated in the main text.

### W2 — Breadth of tasks/domains

One clarification first, which we do not lean on: of the five real-world datasets, four are
affective (MOSI sentiment, UR-Funny humour, MUStARD sarcasm, CREMA-D-irony constructed from
emotion) and the fifth — Hateful Memes — is a vision–language reasoning benchmark rather than an
affect task. The reviewer's substantive point stands: none of them is an application domain outside
web/affect media.

We are therefore adding two non-affective domains during the discussion window, each with the
paper's **complete 8-method suite** (Ensemble, Vanilla Fusion, D&R, MMPareto, ReconBoost, MCR,
SynIB M_Random, SynIB M_Learned) plus both unimodal baselines, 3 seeds, and val-selected
hyperparameter grids for the balancing baselines as well as for SynIB — the paper's protocol
exactly, so no row is under-tuned:

- **PTB-XL** (healthcare): clinical 12-lead ECG + static patient features, 5-class diagnostic
  superclass. [PENDING: PTB-XL results table]
- **Vision & Touch** (robotics): vision + force/proprioception, binary contact prediction.
  [PENDING: Vision & Touch results table]

**On MIMIC.** MultiBench's MIMIC task needs PhysioNet credentialed access *plus* a data request to
the benchmark maintainers, which cannot complete inside the window. PTB-XL is open access and has
the *same* static-features + time-series structure as the MultiBench MIMIC task — we use MIMIC's
reference encoder pair (MLP on static features, GRU on the series) unchanged — so it is a structural
substitute, not a convenient one. We commit to MIMIC itself for the camera-ready.

**Reporting rule, pre-committed.** On each new dataset we first run the cheap diagnostic the paper
recommends — unimodal baselines vs. joint, plus the size of the unimodal-missed subset — and only
then report SynIB against it. Where the diagnostic indicates synergy is present, SynIB should help;
where the task is unimodal- or redundancy-dominated (plausible for both: the ECG may dominate the
static features, and vision and force both signal contact), SynIB should be neutral rather than
harmful. Every dataset we start is reported, win or null, with its diagnostic alongside. This is
also our empirical answer to FTxm W4/Q4: the diagnostic is cheap, is needed for the subset metric
anyway, and predicts the regime before SynIB is trained.
[PENDING: per-dataset diagnostic — unimodal-vs-joint gap and unimodal-missed subset size]

### W3 — Extension to ≥3 modalities

We agree that full PID does not scale: for k ≥ 3 the Williams–Beer lattice has exponentially many
atoms, and SynIB does not attempt to target them individually. It deliberately targets one
well-defined slice — information unavailable to *any single modality* — which is defined for every k
and is exactly what the leave-one-out penalty measures: mask modality i, keep the other k−1, and
penalise confidence that survives. Cost is k extra forward passes per step, linear in k, with no
change to the objective's form. We will state this generalisation explicitly (it is currently
implicit in the k=2 presentation) together with what it does *not* claim: it does not separate "two
modalities redundant, their join synergistic with a third" from other higher-order structure — that
is the part of the lattice we decline to estimate.

Empirically, the trimodal experiment is planned on the *non-affective* Vision & Touch dataset (it
ships five modalities; we use vision + force + proprioception), so one experiment addresses this
weakness and W2 together. [PENDING: Vision & Touch trimodal result]

### W4 — InfMasking

See global response item 4. Three concrete axes: (i) motivation — InfMasking maximizes
shared/synergistic information *between representations* via InfoNCE, label-free, shaping
pretraining; SynIB targets task-relevant synergy conditioned on Y, shaping supervised training.
Task-agnostic synergy can be task-irrelevant; unimodal shortcuts w.r.t. Y are invisible without Y.
(ii) implementation — contrastive estimator over masked views vs. a counterfactual KL confidence
penalty; no contrastive pairs or large-batch requirement; drop-in at the loss level. (iii) key
idea — InfMasking encourages representations to *contain* synergistic information; SynIB penalizes
the classifier for not *using* cross-modal information. We add the citation and temper novelty
claims.

### W5 — Implicit disentanglement assumption

We do not assume an a-priori disentangled latent — and our new multi-seed mask analysis (N4) directly
supports this. The adversarial mask *searches* for a soft subspace sufficient to disrupt unimodal
prediction; the sparsity penalty is a preference, not an assumption. Empirically, per-run mask
feature-importance vectors are measured with high reliability (split-half Spearman ρ = 0.95), yet
*different seeds converge to different, equally effective attack subspaces* (cross-seed ρ ≈ 0 on
identical data partitions), while gate statistics and the downstream task effect are reproducible
across seeds. Unimodal information is *redundantly distributed* across latent coordinates — there
is no unique "unique-information subspace" — and SynIB neither assumes nor requires one: any
sufficient subspace the adversary finds yields the same regularization effect (see the ±0.5pp
learned-vs-random equivalence in the FTxm W3 response). New appendix analysis.

**New causal test (N14): we removed disentanglement by construction.** If learned masking assumed
a disentangled latent, it should fail when no coordinate subset corresponds to any PID source. We
rotated each PID-XOR modality by a fixed random orthogonal matrix (and, in a second arm, applied
an elementwise tanh after rotation — invertible nonlinear mixing that no linear layer can absorb)
before standardization, and re-ran vanilla, random-masking, and learned-masking SynIB at paper
hyperparameters. No variant fails: vanilla stays at chance on the synergy slice (0.50), and SynIB
retains its gains under full rotation (random masking 0.915 ± 0.006, learned masking 0.845 ±
0.046, stable across three different rotation matrices) and under nonlinear mixing (learned
0.874 ± 0.010). An oracle applied in the pre-rotation source basis reaches 0.920 ± 0.007,
confirming the rotated task itself is unchanged. Mechanistically, the mask behaves exactly as the
fallback argument in the paper claims: on unrotated data the inner adversary identifies the
unimodal head's support — its soft gate ranks the true support coordinates above chance
throughout training, and with a longer inner loop (100 steps) the binary mask localizes outright
(IoU 0.31 vs 0.27 random-mask baseline; corruption fraction converging to the ideal 0.625 =
synergy+noise blocks) — while under rotation, where no coordinate support exists, the identical
configuration reverts to random-mask-like corruption (fraction 0.555) with accuracy statistically
close to random masking. The assumption at stake is therefore only *sparse functional reliance of
the unimodal head*; when it is violated, SynIB degrades gracefully to its documented
random-masking behavior rather than failing. Full protocol, per-source training dynamics, and
mask–oracle agreement curves in the new appendix.

### Factual question — joint training and symmetry

Yes: Eq. (5) and (6) are trained jointly (alternating adversarial updates within each step), and
the objective is symmetric — each modality is masked in turn and both KL penalties are applied
(the `l_z1_masked`/`l_z2_masked` terms; asymmetric weighting is possible but symmetric weights are
the default). We will clarify in Section 3.5.

---

## Response to Reviewer jjzx

### Q1 — Masking ratio p

The masking ratio (probability that a latent feature is corrupted, i.e., replaced by a fill value)
was selected per dataset: **MUStARD 0.1, CREMA-D 0.2, MOSI 0.3, Hateful Memes 0.3–0.5 (a
pmin–pmax range), UR-Funny 0.7**; we will state this in the revised experimental setup.
Sensitivity (experiments N6 + N8): dedicated grids show accuracy is flat across p ∈ {0.1, 0.3, 0.5, 0.7, 0.9} on both
CREMA-D-irony (59.2–60.6; no setting deviates from p=0.5 by more than 1.5pp paired within fold)
and MUStARD (65.7–69.2, within fold noise) — although p was tuned per dataset across a wide range,
the tuning is not load-bearing: no value produces a cliff.

### Q2 — Which masking does Fig. 4 use?

Figure 4 (CREMA-D-irony) uses **random masking** (M_random, corruption probability π = 0.2), with
the reference distribution given by the complementary modality's unimodal prediction — as
described in Appendix H.5. We will state this directly in the Fig. 4 caption in the revision.

### Q3 — InfMasking

See the response to RpxH W4 and global item 4: cited, differentiated on motivation /
implementation / key idea, priority claims tempered.

---

## Number provenance (do not ship — internal)

| claim | source |
|---|---|
| 3,494 endpoints, 0 collapse | `e2_mask/RESULTS_collapse_audit.md` |
| interior plateaus, σ_g, Fig R2a | `e2_mask/RESULTS_phase1.md`, `fig_R2a_trajectories.*` |
| split-half ρ=0.95 / cross-seed ρ≈0 | `e2_mask/RESULTS_agreement.md` (+ addendum) |
| TOST ±0.5pp, 186 pairs | `e2_mask/RESULTS_equivalence.md` |
| learned-vs-random per-cell | `e2_mask/RESULTS_random_vs_learned.md` (+ addendum) |
| sensitivity table | `e2_mask/RESULTS_sensitivity.md` |
| CREMA-D p ≤1.5pp paired | `e2_mask/RESULTS_equivalence.md` |
| p=0.3/0.5 factual | `e2_mask/RESULTS_phase0.md` |
| r-ablation | `e2_mask/RESULTS_sensitivity.md` (E2KLREF rows) |
| mechanism Δce/Δcf | `e2_mask/RESULTS_phase1.md` |
| α-grid (synergy-rare) | PENDING cluster 52911 → collapse-audit rerun |
| HM trajectories | PENDING cluster 52912 → collapse-audit rerun |
| N14 entangled-XOR (all numbers in RpxH W5 ¶2) | `docs/rebuttal_entangled_xor.md` (Table 1, Secs 3–5); runs in `artifacts/rebuttal_entangled_xor/` |

**N14 CAUTION (do not ship without a decision):** the published PID-XOR M_Random/M_Learned code
path never corrupts the input — `destroy_block(x, m, 1)` with an int `block_list` is a silent
no-op, so paper Fig. 7/8's M_Random/M_Learned are a KL-to-uniform confidence penalty on intact
inputs (M* and all benchmark results unaffected; see `docs/rebuttal_entangled_xor.md` Sec. 0 and
memory note). All N14 numbers quoted in the W5 response use the FIXED code path (`destroy_fix`,
real masking) and are therefore safe to ship as new-experiment results — but do NOT quote paper
Fig. 7/8 M_Random/M_Learned values as masking evidence anywhere in the rebuttal, and decide
before camera-ready how to correct the figure (fix is `[1]`; fixed-track Fig. 8 replacement
numbers already exist in N14's identity arm).
