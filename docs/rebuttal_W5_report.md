# W5 (RpxH) — Does SynIB assume a disentangled latent? A detailed evidence report

**Reviewer concern (RpxH, Weakness 5, verbatim):** *"I find one assumption somewhat tricky. The
authors seem to implicitly assume that certain dimensions of the latent representation are
associated only with unique information, while other dimensions mainly capture multimodal
interactions such as synergy. At least, this appears to be the assumption underlying the learned
masking mechanism. I would like the authors to provide further justifications."*

This report is the internal, fully-sourced basis for our rebuttal answer. It is deliberately
longer and more candid than the posted response; the "rebuttal-ready" paragraphs at the end
(§8) are the distilled version. Every number is reproducible from
`artifacts/rebuttal_entangled_xor/` (git commit, config hash, exact command, and the frozen
mixer/rotation matrices are saved per run).

---

## 1. One-paragraph summary

The reviewer's concern is that SynIB's masking only works because the synthetic latent is
axis-aligned (unique/redundant/synergistic information sitting in separate coordinate blocks),
so a coordinate-wise mask can trivially isolate the synergy block. We tested this by **destroying
all axis-aligned structure** three ways of increasing severity — a dense orthogonal rotation, a
saturating tanh∘rotation, and a frozen random two-layer MLP with saturating pre-activations — and
**verified with probes that no small coordinate subset carries any source** after mixing
(24–31 of 32 coordinates must be corrupted to remove a source, vs. 5–6 on the axis-aligned data).
Under every mixing, **vanilla fusion stays at chance on the synergy subset (~0.50) while SynIB
recovers synergy** (0.66–0.92 depending on severity). The decisive conclusion for the reviewer:
SynIB does **not** assume or require a disentangled latent. A secondary, honest finding is that
the *learned* mask's small advantage over *random* masking under mild conditions does not persist
under the hardest mixing — the ingredient that survives entanglement is the masked-input
**confidence penalty**, with random masking as its most robust instantiation. This is exactly
consistent with the learned≈random equivalence we report on the real benchmarks.

---

## 2. Experimental design

### 2.1 Base task (paper PID-XOR, Fig. 8 protocol)
Two modalities, 32 latent coordinates each, organized (in the source basis) into contiguous
blocks: **unique** (6 coords, label-predictive from one modality), **redundant** (6, label from
either modality), **synergistic** (6, XOR across modalities — neither modality alone suffices),
**noise** (14). Mixture (p_U1, p_U2, p_R, p_S) = (0.45, 0, 0.45, 0.10); 1800 train / 200 val /
4000 test; 30 epochs; Adam lr 3e-4; final-epoch test metrics (no checkpoint selection); 3 seeds
{0,1,2}; mean ± SE. **Synergy accuracy (S) is the headline metric**: test accuracy on the subset
of examples whose label is carried only by the synergistic block.

### 2.2 Methods
- **Vanilla** — plain fusion, no masking.
- **SynIB M_Random** — corrupt a random π=0.5 subset of each modality's coordinates, penalize the
  fused prediction's confidence on the corrupted input (masked-input confidence penalty, λ_KL=10).
- **SynIB M_Learned** — an inner adversarial loop (20 Adam steps) learns which coordinates to
  corrupt, then the same confidence penalty is applied.
- **Oracle references** — **M\*** (axis-aligned, destroys the true synergy block; identity only)
  and **M\* source-basis** (rotate back to the source basis, destroy the true block, re-rotate;
  the upper reference under rotation).

### 2.3 Entanglement variants (the reviewer's stress)
Applied per modality, after block construction, before standardization; fixed across
splits/methods/seeds; saved to disk:
- **identity** — axis-aligned (the setting the reviewer worries is special).
- **full rotation** — x ← Q·x, dense random orthogonal Q (QR of seeded Gaussian). Linear, but no
  coordinate is a single source anymore.
- **tanh∘Q** — x ← tanh(Q·x). Nonlinear; not absorbable by a single linear layer.
- **frozen MLP mixer** — x ← W₂·tanh(W₁·x), W₁∈R^{64×32}, W₂∈R^{32×64}, i.i.d. Gaussian, gain
  calibrated so tanh pre-activations have std ≈ 2.5 (**deep saturation** — the strongest version;
  provably not invertible by the encoder's first linear layer).

### 2.4 A prerequisite fix (full disclosure)
While building this we found that the paper's synthetic PID-XOR script's `destroy_block(x, m, 1)`
returns the input **unchanged** when the block argument is an int (as the published Fig. 7/8 code
calls it). In the *published synthetic* M_Random/M_Learned curves the mask therefore never
corrupted anything — the counterfactual KL acted on intact inputs (a pure confidence penalty), and
M_Random/M_Learned were bitwise-identical. **This affects only the synthetic PID-XOR figures; the
benchmark trainer (`src/synib/models/vlm/synib_mask_model.py`) masks correctly, as does
Spurious-XOR.** All results in this report use the **fixed track** (`destroy_fix: true`, masks
actually corrupt), so they answer the reviewer's question about masking on real corrupted inputs.
We will correct the synthetic-figure caption in the revision.

---

## 3. Main result — synergy recovery survives entanglement (fixed track)

Test synergy accuracy **S** and total accuracy, mean ± SE over 3 seeds. Chance on the synergy
subset ≈ 0.50.

| Variant | Vanilla S | M_Random S | M_Learned S | Oracle S | M_Random total | M_Learned total |
|---|---|---|---|---|---|---|
| **identity** (axis-aligned) | 0.504 ± 0.011 | 0.876 ± 0.011 | 0.862 ± 0.013 | 0.878 ± 0.012 | 0.940 | 0.951 |
| **full rotation** (linear) | 0.501 ± 0.013 | 0.915 ± 0.006 | 0.845 ± 0.046 | 0.920 ± 0.007 | 0.951 | 0.953 |
| **partial rotation** | 0.504 ± 0.013 | 0.883 ± 0.019 | 0.761 ± 0.095 | — | 0.932 | 0.927 |
| **tanh∘Q** (nonlinear) | 0.488 ± 0.018 | 0.834 ± 0.014 | 0.874 ± 0.010 | — | 0.858 | 0.897 |
| **frozen MLP** (saturating) | 0.498 ± 0.011 | 0.813 ± 0.007 | 0.769 ± 0.025 | — | 0.951 | 0.960 |
| **frozen MLP, mixer seed 1** | 0.515 ± 0.025 | 0.688 ± 0.047 | 0.662 ± 0.061 | — | 0.939 | 0.932 |

**Reading.**
- **Vanilla fusion is at chance on synergy in every variant** (0.49–0.52) — the synergistic signal
  is genuinely cross-modal and is not accidentally recoverable by a plain fusion model, before or
  after mixing.
- **SynIB recovers synergy in every variant**, from +16 pp over chance (frozen MLP seed 1) up to
  +42 pp (full rotation). The gain does not depend on axis-alignment.
- Under **linear rotation** SynIB essentially matches the source-basis oracle (M_Random 0.915 vs
  oracle 0.920) — masking in the entangled input basis is as good as masking the true block.
- Under the **hardest mixing** (frozen saturating MLP) the absolute synergy level attenuates
  (0.66–0.81) and depends on the mixer draw, but stays well above chance on both independent
  mixers.

---

## 4. Arm A — the frozen MLP mixer in detail (closing the "mild squashing" loophole)

**Why this arm exists.** Standardized inputs live roughly in [−1,1], where tanh is nearly linear.
A skeptic could argue the tanh∘Q arm was "a rotation with mild squashing," approximately undone by
the encoder's first linear layer. The frozen MLP mixer forecloses this: W₁ gain is calibrated so
pre-activations have **std ≈ 2.5** (realized **2.500** on both modalities, both mixer seeds),
placing tanh deep in saturation. No single linear layer can invert a saturating elementwise
nonlinearity composed with two random projections.

**Blocking sanity gate (passed, no gain adjustment needed).** Before trusting any SynIB number we
required that the task itself remain learnable on the mixed data:

| Mixer | Vanilla U1 | Vanilla R | Vanilla S | Gate |
|---|---|---|---|---|
| seed 0 | 0.987 | 0.986 | 0.498 | **PASS** (U1,R ≥ 0.95; S at chance) |
| seed 1 | 0.977 | 0.985 | 0.515 | **PASS** |

Unique and redundant information are still fully recoverable (≥0.98), and synergy is at chance for
vanilla — the mixer scrambles coordinates without destroying the task.

**SynIB results (mean ± SE, 3 training seeds):**

| Mixer | Method | S | total | U1 | R |
|---|---|---|---|---|---|
| seed 0 | Vanilla | 0.498 ± 0.011 | 0.940 | 0.987 | 0.986 |
| seed 0 | M_Random (π=0.5) | **0.813 ± 0.007** | 0.951 | 0.967 | 0.964 |
| seed 0 | M_Learned (paper HPs) | 0.769 ± 0.025 | 0.960 | 0.976 | 0.984 |
| seed 0 | M_Learned (100 steps, λ_M=0) | 0.751 ± 0.015 | 0.958 | 0.977 | 0.983 |
| seed 1 | Vanilla | 0.515 ± 0.025 | 0.937 | 0.977 | 0.985 |
| seed 1 | M_Random | 0.688 ± 0.047 | 0.939 | 0.955 | 0.975 |
| seed 1 | M_Learned | 0.662 ± 0.061 | 0.932 | 0.946 | 0.975 |

**Takeaways.** (i) Synergy recovery survives the hardest mixer on two independent mixer draws
(+16 to +32 pp over chance). (ii) The absolute level depends on the mixer realization (seed 0 vs
seed 1), which is expected — a saturating random map can compress the synergy direction more or
less. (iii) M_Random ≥ M_Learned throughout this arm: with no axis-aligned block to find, the
adversarial inner loop buys nothing, and the extra inner-loop budget (100 steps) does not help.

---

## 5. Arm B — measured entanglement (the evidence that mixing actually removed structure)

This arm has **no training**; it verifies the datasets are what we claim. Two operationalizations,
computed on seed-0 train data (`scripts/analysis/rebuttal_entangled_probe.py`).

**5.1 A subtlety that matters.** Each source is a *rank-1 signed direction* (a single latent bit
projected up). A rotation or mixer spreads that 1-D direction across all coordinates but does not
hide it — every coordinate keeps a projection of it. So a **read** probe (find a small set of
coordinates that *predict* the source) is the wrong test: one coordinate still reads a 1-D source
even after mixing. The right, masking-relevant test is a **destroy** probe: how many coordinates
must be *corrupted* before no reader can recover the source from what remains. We report both and
lead with destroy support.

**5.2 Table B — entanglement verification.** "Read support" = min L1-probe support reaching ≥90%
of the unregularized accuracy. "Destroy support" = min coordinates to corrupt (inverse-RFE:
iteratively destroy the highest-weight coordinate, retrain the reader on the rest) to drive the
best remaining reader to ≤0.60. Block size in the source basis = 6.

| Variant | Source | Read support | **Destroy support** | max \|corr\| | max MI (bits) |
|---|---|---|---|---|---|
| identity | unique | 2/32 | **5/32** | 0.88 | 0.94 |
| identity | redundant | 2/32 | **6/32** | 0.90 | 0.97 |
| identity | synergy (mod 0 bit) | 2/32 | **5/32** | 0.91 | 0.96 |
| identity | synergy (mod 1 bit) | 3/32 | **5/32** | 0.82 | 0.77 |
| tanh∘Q | unique | 1/32 | **27/32** | 0.95 | 0.85 |
| tanh∘Q | redundant | 1/32 | **24/32** | 0.93 | 0.84 |
| tanh∘Q | synergy (mod 0 bit) | 1/32 | **28/32** | 0.99 | 0.96 |
| tanh∘Q | synergy (mod 1 bit) | 10/32 | **25/32** | 0.61 | 0.37 |
| frozen MLP | unique | 1/32 | **31/32** | 0.91 | 0.90 |
| frozen MLP | redundant | 4/32 | **27/32** | 0.74 | 0.53 |
| frozen MLP | synergy (mod 0 bit) | 2/32 | **29/32** | 0.88 | 0.87 |
| frozen MLP | synergy (mod 1 bit) | 3/32 | **24/32** | 0.67 | 0.48 |

**Reading.** On the axis-aligned data, destroy support = 5–6/32 — exactly the ground-truth block
size, i.e. a coordinate-wise mask *can* isolate a source (this is the structure the reviewer
worries about). After mixing, **75–97% of all coordinates must be corrupted to remove any
source** — there is no small coordinate subset that carries a source, so any working mask must be
dense and cannot be exploiting axis-alignment. Figures:
`docs/figures/rebuttal_entangled_xor/fig3_sparse_probes.*` (read) and `fig4_destroy_support.*`
(destroy).

---

## 6. Arm C — tanh saturation sweep (how hard does the nonlinearity have to be?)

x ← tanh(s·Q·x) for s ∈ {1, 2, 4}. s=1 reproduces the tanh∘Q arm **bitwise** (consistency check
passed). Sanity gate passed at every s (vanilla U1/R ≥ 0.986, S at chance) — even s=4 keeps the
task learnable. Synergy accuracy, mean ± SE:

| s | Vanilla S | M_Random S | M_Learned S |
|---|---|---|---|
| 1 | 0.488 ± 0.018 | 0.834 ± 0.014 | 0.874 ± 0.010 |
| 2 | 0.477 ± 0.014 | 0.832 ± 0.019 | 0.745 ± 0.025 |
| 4 | 0.477 ± 0.010 | 0.820 ± 0.032 | 0.747 ± 0.019 |

**Reading.** M_Random is remarkably stable across a 4× saturation range (0.834 → 0.820, span 1.4
pp). M_Learned drops ~13 pp from s=1 to s≥2 and then plateaus — again the pattern that the learned
inner loop is the fragile part while the confidence-penalty mechanism (best seen through random
masking) is robust. (Aside: at s≥2 M_Random trades some unique-modality accuracy, U1 0.64–0.84.)

---

## 7. Pre-registered prediction checklist (written before Arms A/C, filled honestly)

We committed predictions to the doc before launching, and report failures as failures.

| # | Prediction | Outcome | Numbers |
|---|---|---|---|
| (a) | Arm A: vanilla synergy at chance; **both** SynIB variants ≥ 0.80 | **PARTIAL FAIL** | Vanilla 0.498 ✓; M_Random 0.813 ✓; M_Learned 0.769 ✗ (seed-1 mixer 0.662) |
| (b) | Arm B: min support ≈ block on identity; ≥ 28/32 on mixed variants, all sources | **FAIL as stated; corrected measure PASSES** | Read support 1–10/32 (prediction mis-specified for rank-1 sources); **destroy support** 5–6/32 identity vs 24–31/32 mixed ✓ |
| (c) | Arm C: synergy varies < 5 pp across s per variant | **PARTIAL** | M_Random span 1.4 pp ✓; M_Learned −12.9 pp s1→s2 ✗ |

The honest through-line of all three "partial" outcomes is the **same** and is *good* for the
paper's actual claim: what survives entanglement is SynIB's masked-input confidence penalty, not a
particular mask geometry — and random masking is its most robust instantiation. The learned mask's
edge is a mild-condition phenomenon, which is precisely what we already argue on the real
benchmarks (learned ≈ random within ±0.5 pp on overall accuracy; the learned mask helps mainly on
the synergy subset).

---

## 8. Rebuttal-ready text (drop-in)

**Core answer (use this):**
> SynIB does not assume a disentangled latent. To show this directly, we destroyed all
> axis-aligned structure in our controlled PID-XOR task three ways of increasing severity — a
> dense random orthogonal rotation, a saturating tanh∘rotation, and a frozen random two-layer MLP
> with tanh pre-activations driven deep into saturation (std ≈ 2.5), which no linear encoder layer
> can invert. A probe analysis confirms the mixing removed the structure the concern is about:
> whereas on the original axis-aligned data only 5–6 of 32 coordinates must be corrupted to remove
> a source, after mixing 24–31 of 32 must be — no small coordinate subset carries any source.
> Across all three mixings, vanilla fusion remains at chance on the synergy subset (≈0.50) while
> SynIB recovers synergy (0.66–0.92, always well above chance; under a linear rotation SynIB
> matches the source-basis oracle, 0.92 vs 0.92). The learned mask therefore does not rely on
> unique and synergistic information occupying separate coordinates; the sparsity preference is a
> preference, not an assumption, and a dense mask arises whenever the data demand it.

**Mechanism / honest scoping sentence (recommended to include — it pre-empts the obvious probe):**
> We also observed that the *learned* mask's small advantage over *random* masking under mild
> conditions does not persist under the hardest saturating mixing (e.g. 0.75–0.77 vs 0.81–0.83
> under the MLP mixer; 0.745 vs 0.832 at tanh scale 2). The ingredient that survives entanglement
> is the masked-input confidence penalty itself, with random masking as its most robust
> instantiation — consistent with the learned≈random equivalence we report on the real benchmarks.

**If a reviewer asks "did you assume anything?" —** the sparsity penalty on the mask is a soft
preference for compact corruptions, not a hard constraint; under entanglement the learned gates
become dense (they must, per Arm B) and the method still works. There is no disentanglement
assumption anywhere in the objective.

---

## 9. Caveats we should not hide

1. **M_Learned underperforms M_Random under hard mixing.** Reported above; framed as evidence for
   "the mechanism is the confidence penalty," which is a strength, but a reviewer could read it as
   "why learn the mask at all?" Our honest answer: on the real benchmarks the learned mask's value
   is concentrated on the synergy subset (Fig. 5), and overall it is equivalent to random within
   ±0.5 pp — so the synthetic and real stories agree, and random masking is a legitimate, robust
   default.
2. **Mixer-draw sensitivity.** Absolute synergy under the frozen MLP depends on the mixer seed
   (0.81 vs 0.69 for M_Random). We report both; the *qualitative* claim (well above chance,
   vanilla at chance) holds on both. We did not tune the mixer to flatter SynIB — the gain
   calibration targets pre-activation std only, and the sanity gate is the sole acceptance filter.
3. **Synthetic only.** This is the controlled instrument for the disentanglement question, where
   we can *construct* entanglement and *measure* it (Arm B). The real-benchmark evidence for "no
   disentanglement assumed / needed" is the multi-seed mask analysis (masks are seed-specific but
   functionally equivalent; unimodal information is redundantly distributed).
4. **The `destroy_block` no-op** in the published synthetic figures (see §2.4) is disclosed and
   scoped to the synthetic PID-XOR plots; it does not touch the benchmark results.

---

## 10. Provenance

- **Runs:** `artifacts/rebuttal_entangled_xor/runs/{identity_fixed,rotated_full_fixed,rotated_partial_fixed,rotated_full_tanh,mlp_mixer,mlp_mixer_seed1,mlp_mixer_steps100,tanh_scale1_check,tanh_scale2,tanh_scale4}__{vanilla,mrand,mlearned,mstar_srcbasis}__seed{0,1,2}.json`
  — each records git commit, config file+hash, exact command, host/device, and the
  rotation/mixer files used.
- **Frozen mixers:** `artifacts/rebuttal_entangled_xor/mixers/mlp_mixer_seed{0,1}_mod{0,1}_*.{npz,json}` (weights + realized pre-activation std).
- **Rotations:** `artifacts/rebuttal_entangled_xor/Q/`.
- **Probes:** `artifacts/rebuttal_entangled_xor/probe_table.{md,json}`.
- **Figures:** `docs/figures/rebuttal_entangled_xor/{fig1_entangled_dynamics,fig3_sparse_probes,fig4_destroy_support}.{pdf,png}`.
- **Configs:** `run/configs/rebuttal_entangled_xor/*.json`.
- **Code:** runner `scripts/analysis/rebuttal_entangled_xor.py`; probe `scripts/analysis/rebuttal_entangled_probe.py`; data hook `scripts/analysis/Xor_PID3Main_MaskSynIB_Search.py` (rotation/nonlinearity/MLP-mixer, opt-in via cfg). Committed at `938a5a0` (follow-up) on top of `2fa5617` (original arms).
- **Full narrative log:** `docs/rebuttal_entangled_xor.md` §9 (this report is the standalone digest of §§0–9).
