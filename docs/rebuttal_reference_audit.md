# Reference-distribution audit (Rebuttal — Reviewer RpxH, W1)

**Question.** Which reference distribution `r` did each paper run actually use in the SynIB KL
penalty `D_KL(q(·|x̃_i, x_{-i}) ‖ r)`? App. C.2 of the paper claims the uniform reference
"in all experiments"; App. H.5 claims the complementary modality's unimodal prediction on
CREMA-D. These cannot both be right. Answer determined from configs + code only (paper-release
repo `SynIB`, commit `3f3d3b8`), not from docs.

## Verdict table

The paper command for each dataset is the `M_random` SynIB command in `docs/REPRODUCE.md`.
"Reference actually used" is what the loss code computes for that command.

| Dataset | Config (`--config`) | Model class | `synergy_type` resolved | KL form | Reference `r` actually used | π actually applied | λ |
|---|---|---|---|---|---|---|---|
| UR-Funny | `run/configs/multibench/urfunny/synib.json` | `FusionIBModel_Mask` | *absent* → default `"gaussian"` | Gaussian KL of fused **logits** vs `N(0, I)` (App. C.1 family) | `N(0, I)` over logits — *not* categorical-uniform | 0.7 | 0.001 |
| MUStARD | `run/configs/multibench/mustard/synib.json` | `FusionIBModel_Mask` | *absent* → default `"gaussian"` | same as UR-Funny | `N(0, I)` over logits | 0.1 | 0.001 |
| MOSI | `run/configs/multibench/mosi/synib_u.json` | `FusionIBModel_Mask_U` | `"unimodal_anchor"` | categorical `KL(q_masked ‖ softmax(uni_pred))` | **the masked modality's own clean unimodal prediction**, live co-trained head, detached | 0.3 | 0.1 |
| Hateful Memes | `run/configs/hateful_memes/methods/synib_u.json` (merged over `tiers/small_tf_deberta.json`) | `FusionIBModel_Mask_U` | `"unimodal_anchor"` (method overrides tier's `"gaussian"`) | same as MOSI | **a randomly-initialized, never-trained linear head** (see Finding 3) | 0.3 (fixed — `p_max` unused, see Finding 5) | 0.01, `l_pareto` 0.1 |
| CREMA-D-Irony | `run/configs/cremad/synib_u.json` | `FusionIBModel_Mask_U` | `"unimodal_anchor"` | same as MOSI | masked modality's own clean unimodal prediction, live co-trained head (pretrained-initialized encoders), detached | **0.20** (not 0.3) | 1.0 |

Code paths for the two KL forms (all datasets resolve to the single implementation in
`src/synib/models/vlm/synib_mask_model.py`; `Loader._resolve_model_class` at
`src/synib/training/pipeline/helpers/Loader.py:39` resolves through `src/synib/models/__init__.py`,
and `crema_d/` defines no FusionIB classes — CREMA-D uses the same file):

- `synergy_type` default: `synib_mask_model.py:582` — `getattr(args, "synergy_type", "gaussian")`.
- Gaussian form: `_gaussian_kl` (`:633-635`), applied by `_kl_loss`/`_kl_pass` (`:1089-1099`) with
  `mu` = the masked-branch **fused prediction logits** and `logvar` from `logvar_head(feat)`.
  Minimized at logits = 0, i.e. uniform softmax at the mode, but it is a Gaussian prior on logits,
  **not** the categorical `KL(π‖ρ)` of App. C.2 Eq. (14).
- Anchor form: `_kl_unimodal_anchor` (`:1101-1106`) — `KL(log_softmax(pred_masked) ‖ softmax(target.detach()))`.
- Branch wiring (random branch `:1297-1313`, learned branch `:1315-1330`): both call sites pass the
  targets described in Finding 2.

## Findings

### 1. App. C.2's "uniform in all experiments" is false for every real-data benchmark

No real-data paper run uses the categorical-uniform reference of App. C.2. UR-Funny and MUStARD
use the App. C.1 **Gaussian** family on the fused logits (`r = N(0, I)`); MOSI, Hateful Memes and
CREMA-D-Irony use the **unimodal-anchor** categorical KL. The categorical-uniform reference is used
only in the synthetic XOR experiments (App. F.1: `Bernoulli(0.5)` reference, `docs/paper.txt:1263`,
`:1362`). App. H.5's description (anchor on CREMA-D) matches the code; App. C.2's parenthetical
("we use ρ_k = 1/K … in all experiments", `docs/paper.txt:1065-1066`) does not.

**Consequence for the ablation:** there is no existing "uniform" code path to *verify* on the real
benchmarks — the `uniform` arm of the ablation is a new (but trivial) categorical-KL path, and on
UR-Funny/MUStARD even the paper baseline is a different estimator family (Gaussian-on-logits). The
ablation therefore compares all three references inside one categorical-KL estimator; the paper's
Gaussian runs are quoted as context, not as the "uniform" arm.

### 2. The anchor points at the WRONG modality relative to App. H.5's description

App. H.5: reference is "the complementary modality's unimodal prediction, so the KL term penalizes
the fused prediction for departing from what the **unmasked** modality alone would predict"
(`docs/paper.txt:1576-1578`).

Code: in `_base_forward_synib`, the z2-destroyed branch (`pred_randmask0` = fused prediction with
z1 clean, z2 corrupted, `:1299`) is anchored to `uni_pred_2` — the **masked** modality's own
unimodal prediction on its *clean* input (`:1309`); symmetrically the z1-destroyed branch anchors to
`uni_pred_1` (`:1310`); the learned branch does the same (`:1326-1327`). The same direction exists in
the internal repo's full history (it was never `uni_pred_1`), so all paper `_U` runs (MOSI, HM,
CREMA-D) trained with the masked-modality anchor: "predict from x_{-i} what the destroyed modality
would have said", a cross-modal distillation target — not the `p(·|x_{-i})` reference of the
rebuttal's error decomposition, and not what H.5 describes.

**Consequence:** the ablation's `unimodal_anchor` arm implements the *complementary* (unmasked)
head per H.5 and the task spec. The legacy direction is kept available as `anchor_legacy` for a
control if needed.

### 3. On Hateful Memes the "anchor" is a random, never-trained head

The HM unimodal heads are `nn.Linear(d_model, num_classes)` created at init
(`src/synib/models/vision_text/hf_text.py:43`, `clip_vision.py` analogous) with **no pretrained
checkpoint** (tier config `pretrainedEncoder: null`) and **no CE training signal**: the merged HM
config has `multi_loss.multi_supervised_w = {combined: 1, c: 0, g: 0}` and unimodal-head inputs are
detached by default (`detach_unimodal_pred=True`, `synib_mask_model.py:1215-1217`;
`hf_text.py:123-124`). The KL target is also detached (`:1104`). So the HM paper runs' reference is
softmax of a frozen random linear head over projection features that drift during training. On MOSI
and CREMA-D the heads DO co-train (CE weights `c=1, g=1`) — the reference is live (detached each
step), not frozen and not EMA.

### 4. Answer to "frozen or EMA?" for the paper runs

Neither. MOSI/CREMA-D: live co-trained head, gradient-detached at the KL call. HM: frozen-at-random-init
head. The ablation's `unimodal_anchor` arm will use an **EMA copy (decay 0.99), detached** — recorded
here per the task spec — and on HM the anchor heads additionally receive a unimodal CE probe term or
pretrained unimodal-head initialization so the anchor is meaningful (exact choice recorded in the
implementation section of the ablation doc).

### 5. π semantics: `p_max` is dead in every random-mask paper run; two π values in the plan were wrong

`--rmask random` sets `perturb.type="random"` (`train.py:448-451`, `:129-133`). In
`get_random_mask_multiclass`, `p_type != "diff"` selects `make_tilde_once` (`:747`), which corrupts
each feature independently with probability `self.p = perturb.p_min` (`:603`, `:728-735`).
**`p_max` is never read on this path.** Therefore:

- Hateful Memes: REPRODUCE.md passes `--perturb_pmin 0.3 --perturb_pmax 0.5`; the *effective* π is a
  fixed 0.3. The "0.3–0.5 per-batch range" (and the plan's guardrail about `p_max` 0.7 vs 0.5) is
  moot at run time — the config-vs-doc drift (`methods/synib.json` has `p_max: 0.7`) is real but has
  no effect. The ablation passes `--perturb_pmin 0.3 --perturb_pmax 0.5` exactly as REPRODUCE.md does.
- CREMA-D-Irony: the paper command uses `--perturb_pmin 0.20` (`docs/REPRODUCE.md:155`), not 0.3 as
  the experiment plan assumed. The ablation uses **0.20**.
- Confirmed π per dataset: UR-Funny 0.7, MUStARD 0.1, MOSI 0.3, HM 0.3, CREMA-D 0.20.

### 6. `--perturb_fill ema` is dead code on the random branch

`fill_func` (`synib_mask_model.py:723-726`) replaces corrupted coordinates with **in-batch shuffled
features** (`eps = z[randperm]`); the `noise_fn`/EMA fill is bypassed (commented-out line kept in
source). The learned branch (`_apply_destroy`, `:819-837`) does honor `fill`. So all random-mask
paper runs corrupt by feature-swap within the batch regardless of the `ema` flag.

### 7. Both mask branches are active in every SynIB run

`synib_use_random_ce` and `synib_use_learnable_kl` both default `True` (`:1286-1287`) and nothing in
the paper commands disables them. Every λ>0 run therefore executes the random-mask branch AND the
learned-mask inner loop, and applies the KL to both branches' masked predictions
(`kl_synergy_rand_{1,2}` and `kl_synergy_{1,2}`; the trainer sums everything in `output["losses"]`,
`Trainer.py:152-160`). "M_random" vs "M_learned" paper rows differ only in hyperparameters (λ,
`p_min` vs `lsparse`), not in which branches run. The ablation replicates the M_random command
verbatim, so its runs also carry both branches — reference_type plugs into both call sites.

### 8. Seed bookkeeping

`train.py:41-44` maps folds 0/1/2 → seeds 109/19/337 for all these datasets. REPRODUCE.md's claim
that "the paper HM seeds are 109 / 27 / 3407" (`docs/REPRODUCE.md:140`) does not match the code —
flagging as doc drift; the ablation uses the code's fold→seed map.

### 9. Prior art in the internal repo

`synib_internal` commit `95b4102` (rebuttal E8) already added a minimal `perturb.kl_ref ∈
{anchor, uniform, marginal}` hook inside `_kl_unimodal_anchor` only. It does not cover the Gaussian
path, its "marginal" is a per-batch mean of the anchor predictions (drifts; not the fixed empirical
class prior this ablation requires), and it keeps the legacy anchor direction. The Step-1
implementation supersedes it with `reference_type ∈ {uniform, class_prior, unimodal_anchor}` while
leaving `kl_ref` untouched for E8 reproducibility.

## Where the ablation runs

The paper-release repo has no `artifacts/` (caches, CEUs, pretrained unimodal encoders).
All runs launch from `synib_internal` (143 GB of artifacts, same model code plus rebuttal hooks);
configs for this ablation live in `run/configs/rebuttal_ref_ablation/` there. This doc is the
authoritative audit and is mirrored in both repos.
