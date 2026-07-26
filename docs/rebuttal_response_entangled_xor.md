# Entangled-XOR Response (N14) — Reviewer RpxH, Weakness 5

Standalone, OpenReview-ready response for the disentanglement-assumption concern, built on the
entangled PID-XOR experiment. Companion to `rebuttal_e2_masking_response.md` (which answers the
same weakness from the multi-seed-mask angle, N4) and to the full experimental record in
`docs/rebuttal_entangled_xor.md`.

Format: reviewer's verbatim text, then a short ship version and a full version, then the evidence
tables and internal notes. All numbers final; every run recorded in
`artifacts/rebuttal_entangled_xor/`.

**How this composes with the N4 answer already drafted.** N4 argues *no unique unique-information
subspace exists* (seeds find different, equally effective attack subspaces). N14 is the causal
complement: we *remove* axis-aligned structure by construction and show the method still works.
Ship them together — N4 first (premise refuted), N14 second (premise removed experimentally) — or
ship N14 alone if length forces a cut, since it is the stronger, direct test.

---

## Reviewer RpxH — Weakness 5 (verbatim)

> Finally, I find one assumption somewhat tricky. The authors seem to implicitly assume that
> certain dimensions of the latent representation are associated only with unique information,
> while other dimensions mainly capture multimodal interactions such as synergy. At least, this
> appears to be the assumption underlying the learned masking mechanism. I would like the authors
> to provide further justifications.

---

## Short version (ship if length-constrained, ~230 words)

We tested this assumption by removing it from the data. On the PID-Controlled XOR task, where each
PID source occupies a known block of coordinates, we mixed each modality with (i) a fixed random
orthogonal matrix and (ii) a **frozen random two-layer MLP with saturating pre-activations**
(std ≈ 2.5), which no linear layer can absorb. We verified the intended effect rather than assuming
it: a destruction-probe analysis shows that on the original data 5–6 of 32 coordinates must be
corrupted to remove a source, whereas under mixing **24–31 of 32** are required — no small
coordinate subset carries any source.

SynIB does not fail. Vanilla fusion stays at chance on the synergy slice in every mixed variant
(0.48–0.52), while SynIB retains large synergy gains: under rotation 0.915 ± 0.006 (random
masking) and 0.845 ± 0.046 (learned masking), and under the frozen MLP mixer 0.813 ± 0.007 and
0.769 ± 0.025 respectively — reproduced on a second independent mixer draw (0.688 / 0.662 vs
vanilla 0.515). An oracle applied in the pre-mixing basis reaches 0.920 ± 0.007, confirming the
mixed task itself is unchanged.

The assumption SynIB actually needs is weaker: *sparse functional reliance of the unimodal head*.
When even that is violated, the adversary's mask degrades toward random masking — the documented
fallback — rather than breaking. New appendix section.

---

## Full version

We do not assume an a-priori disentangled latent, and rather than argue this we tested it two
ways: by measuring what the mask does (N4, in our FTxm W3 / RpxH W5 mask analysis) and by
**removing axis-aligned structure from the data by construction** (N14, below).

**Setup.** On PID-Controlled XOR each PID source (unique, redundant, synergistic) is written into
a known disjoint block of 6 coordinates out of 32 per modality — the most favourable possible case
for a "disentangled dimensions" reading. We mix each modality immediately after data generation
and before standardization, with a fixed transform shared across splits, methods, and seeds:

1. **Rotation:** a random orthogonal Q per modality (QR of a seeded Gaussian).
2. **Nonlinear mixing:** elementwise tanh after rotation, at saturation scales s ∈ {1, 2, 4}.
3. **Frozen random MLP mixer (primary):** x ← W₂·tanh(W₁·x), W₁ ∈ R^{64×32}, W₂ ∈ R^{32×64},
   i.i.d. Gaussian, with the W₁ gain calibrated so tanh pre-activations have std ≈ 2.5 (realized
   2.500). This is deep in the saturating regime and cannot be absorbed by the encoder's first
   linear layer.

**We verified the mixing achieved its purpose.** Rather than assume entanglement, we measured it
with a destruction probe: iteratively destroy the most informative coordinate and retrain a reader
on the remainder, counting how many coordinates must be destroyed before the best remaining reader
falls to chance-ish accuracy (≤ 0.60). On the original data this recovers the ground-truth block
size exactly (5–6 of 32 per source); under tanh∘Q it is 24–28 of 32, and under the frozen MLP
mixer 24–31 of 32. Every source is spread over essentially the whole coordinate set.

**Result: no variant of SynIB fails.** Vanilla fusion remains at chance on the synergy slice in
every mixed condition (0.477–0.515), confirming the task still requires cross-modal integration,
while unique and redundant accuracy stay ≥ 0.95, confirming the mixed inputs remain learnable.
SynIB's synergy accuracy:

| condition | vanilla | SynIB M_Random | SynIB M_Learned |
|---|---|---|---|
| original (identity) | 0.504 ± 0.011 | 0.876 ± 0.011 | 0.862 ± 0.013 |
| random rotation Q | 0.501 ± 0.013 | 0.915 ± 0.006 | 0.845 ± 0.046 |
| tanh(Q·x) | 0.488 ± 0.018 | 0.834 ± 0.014 | 0.874 ± 0.010 |
| tanh(4·Q·x) | 0.477 ± 0.010 | 0.820 ± 0.032 | 0.747 ± 0.019 |
| frozen MLP mixer | 0.498 ± 0.011 | 0.813 ± 0.007 | 0.769 ± 0.025 |
| frozen MLP mixer, 2nd draw | 0.515 ± 0.025 | 0.688 ± 0.047 | 0.662 ± 0.061 |

The learned-masking result is stable across three independent rotation matrices (0.845, 0.799,
0.839; pooled 0.828 ± 0.024), and an oracle mask applied in the pre-mixing basis reaches
0.920 ± 0.007 under rotation, confirming that mixing leaves the underlying task intact.

**What the mask actually does.** The mechanism behaves exactly as the fallback argument in the
paper claims. On unmixed data the inner adversary *does* find the unimodal head's support: its
soft gate ranks the true support coordinates above chance throughout training, and with a longer
inner loop (100 steps) the binary mask localizes outright — IoU 0.31 against the ground-truth
support versus 0.27 for a random mask, with the corruption fraction converging to 0.625, exactly
the ideal complement of the support. Under mixing, where no coordinate support exists, the
identical configuration reverts to random-mask-like corruption (fraction 0.555) with accuracy
statistically close to random masking. So the assumption at stake is not "dimensions are
disentangled" but the weaker *sparse functional reliance of the unimodal head*; where that holds
the mask exploits it, and where it fails the method degrades gracefully to random masking rather
than breaking.

**Honest scoping.** Under the hardest mixing, learned masking's small edge over random masking
does not persist (0.75–0.77 vs 0.81–0.83; and 0.662 vs 0.688 on the second mixer draw). What
survives entanglement is SynIB's core mechanism — the masked-input confidence penalty — with
random masking as its most robust instantiation. This is consistent with, and independent
evidence for, the learned-vs-random equivalence we report on the real datasets (FTxm W3).

**Changes to the paper.** New appendix section with the full protocol, per-source training
dynamics under mixing, the destruction-probe entanglement table, and the mask–support agreement
curves; Section 3 states the assumption explicitly as sparse functional reliance of the unimodal
head, with the random-masking fallback.

---

## Evidence tables (for the appendix / reviewer follow-ups)

**Measured entanglement (destruction probe, seed-0 train data).** Minimal number of coordinates
that must be destroyed to drive the best remaining-coordinate reader to ≤ 0.60 accuracy:

| source | identity | tanh∘Q | frozen MLP |
|---|---|---|---|
| unique (m0) | 5/32 | 27/32 | 31/32 |
| redundant (m0) | 6/32 | 24/32 | 27/32 |
| synergy bit b0 (m0) | 5/32 | 28/32 | 29/32 |
| synergy bit b1 (m1) | 5/32 | 25/32 | 24/32 |

Ground-truth block size is 6, recovered on identity. (A *read*-support probe is the wrong measure
here: each source is a rank-1 signed direction, so after mixing every coordinate stays
individually correlated with the bit — 1–10 coordinates suffice to read it. Reading was never the
hard part; masking is about destruction.)

**Mask–support agreement (unmixed data, per epoch, held-out batch).**

| | paper HPs (20 steps) | 100-step inner loop | random mask |
|---|---|---|---|
| hard-mask IoU with support | 0.001 | **0.31** | 0.274 |
| soft-gate AUROC for support | 0.56–0.59 (0.74 at epoch 0) | 0.62 | 0.50 |
| corruption fraction | 0.997 | **0.625** (ideal) | 0.50 |

At paper hyperparameters the 20-step budget saturates the gate, so the binarized mask is
uninformative even though the underlying soft gate ranks the support above chance; the 100-step
arm shows the localization is real when the inner loop is given budget.

**Protocol.** PID mixture (p_U1, p_U2, p_R, p_S) = (0.45, 0, 0.45, 0.10); 1800 train / 200 val /
4000 test; 30 epochs; paper hyperparameters throughout (λ_KL = 10, λ_M = 1.0, inner loop 20 Adam
steps at lr 0.1, τ = 1.0); 3 seeds (8 where noted), final-epoch test accuracy, mean ± SE. The
100-step and λ_M-swept arms are reported as clearly labelled ablations, never folded into headline
numbers.

---

## Internal notes (do not ship)

1. **Blocking caveat on provenance.** The published PID-XOR M_Random/M_Learned code path never
   corrupts its input: `destroy_block(x, m, 1)` with an int `block_list` is a silent no-op, so
   paper Fig. 7/8's M_Random/M_Learned curves are a KL-to-uniform confidence penalty on intact
   inputs. M\*, Spurious-XOR, and all benchmark results are unaffected. **Every number in this
   response uses the fixed code path** (`destroy_fix: true`, real masking) and is safe to ship as
   a new-experiment result — but do not quote paper Fig. 7/8 M_Random/M_Learned values as masking
   evidence anywhere, and decide before camera-ready how to correct the figure (fix is `[1]`;
   replacement identity-arm numbers already exist in `docs/rebuttal_entangled_xor.md`).

2. **Pre-registered checklists were filled honestly** (`docs/rebuttal_entangled_xor.md` §5 and
   §9.4), including two partial FAILs: learned masking missed the pre-registered ≥ 0.80 band under
   the frozen mixer (0.769; second draw 0.662), and lost ~13 pp between tanh scales s = 1 and
   s = 2. The shipped claim is therefore "gains survive, attenuated," never "gains unchanged."

3. **Do not describe the rotation arms as showing masking *improves* under entanglement.** Rotated
   arms score ~4 pp higher than identity for every KL-based variant including a no-masking control,
   with vanilla unmoved — a representation-level optimization effect, not a masking effect
   (`docs/rebuttal_entangled_xor.md` §3.1).

4. **Cells at n=3** (learned masking, identity/rotated primary; all mixer and tanh arms). The
   cheap cells were expanded to n=8. If a quoted learned-masking number becomes load-bearing in
   discussion, expand it first (~30 min/cell). The partial-rotation learned cell
   (0.761 ± 0.095, one 0.58 seed) is deliberately not quoted here.

5. **Sources.** Full record `docs/rebuttal_entangled_xor.md`; runs, mixers, probe tables and Q
   matrices under `artifacts/rebuttal_entangled_xor/`; runner
   `scripts/analysis/rebuttal_entangled_xor.py`; probes
   `scripts/analysis/rebuttal_entangled_probe.py`; configs
   `run/configs/rebuttal_entangled_xor/`.
