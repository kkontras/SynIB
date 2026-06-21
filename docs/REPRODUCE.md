# Reproducing the paper results

Every result is produced by the same entry point:

```bash
PYTHONPATH=src python -m synib.entrypoints.train \
    --config <method-config.json> \
    --default_config <dataset-default-config.json> \
    --fold <k> \
    [method flags ...]
```

- `--config` is the **method** config (objective: SynIB / a baseline / a unimodal).
- `--default_config` is the **dataset** base config (optimizer, scheduler, data paths).
- `--fold k` selects the run; we report **mean ± std over folds 0, 1, 2** (three seeds).
- Extra flags override config fields (e.g. `--l`, `--rmask`, `--lr`).

The `run/<dataset>/train.sh` wrappers just set `PYTHONPATH` and fill in `--default_config`
for you, so `./run/multibench/train.sh urfunny-vt <config> <flags>` is identical to the raw
command above. Both forms are shown below.

Before running, build the feature cache once per dataset
(`./run/multibench/download_and_build_cache.sh <ds>`, `./run/hateful_memes/build_cache.sh`). Evaluate a
trained run with the matching `show.sh`. The reported target numbers for every cell are
in the paper ([`paper.pdf`](paper.pdf), Fig. 5 and the result tables).

---

## MOSI / UR-Funny / MUStARD (frozen MultiBench V+T features)

Default (dataset) configs:

| target | `--default_config` |
|---|---|
| `mosi-vt` | `run/configs/multibench/mosi/default.json` |
| `urfunny-vt` | `run/configs/multibench/urfunny/default.json` |
| `mustard-vt` | `run/configs/multibench/mustard/default.json` |

Run each command for `--fold 0`, `--fold 1`, `--fold 2`.

### UR-Funny — SynIB

```bash
# M_random
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/urfunny/synib.json \
    --default_config run/configs/multibench/urfunny/default.json \
    --fold 0 --rmask random --l 0.001 --perturb_pmin 0.7 --perturb_fill ema --lr 0.001 --wd 0.001

# M_learned
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/urfunny/synib.json \
    --default_config run/configs/multibench/urfunny/default.json \
    --fold 0 --rmask learned --l 1 --perturb_lsparse 0.01 --perturb_fill ema --lr 0.001 --wd 0.001
```

### MUStARD — SynIB

```bash
# M_random
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/mustard/synib.json \
    --default_config run/configs/multibench/mustard/default.json \
    --fold 0 --rmask random --l 0.001 --perturb_pmin 0.1 --perturb_fill ema --lr 0.0005 --wd 0.001

# M_learned
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/mustard/synib.json \
    --default_config run/configs/multibench/mustard/default.json \
    --fold 0 --rmask learned --l 1 --perturb_lsparse 10 --perturb_fill ema --lr 0.0005 --wd 0.001
```

### MOSI — SynIB (asymmetric `_U`, batch size 32)

```bash
# M_random
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/mosi/synib_u.json \
    --default_config run/configs/multibench/mosi/default.json \
    --fold 0 --rmask random --l 0.1 --perturb_pmin 0.3 --perturb_fill ema --lr 0.0005 --wd 0.001 --batch_size 32

# M_learned
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/multibench/mosi/synib_u.json \
    --default_config run/configs/multibench/mosi/default.json \
    --fold 0 --rmask learned --l 100 --perturb_lsparse 0.1 --perturb_fill ema --lr 0.0005 --wd 0.001 --batch_size 32
```

### Baselines + unimodals (all three datasets)

Same `--config`/`--default_config` pattern; just swap the method config and keep
`--fold {0,1,2}` (HP are baked into each config). Example for UR-Funny:

```bash
B=run/configs/multibench/urfunny ; D=$B/default.json
PYTHONPATH=src python -m synib.entrypoints.train --config $B/ensemble.json        --default_config $D --fold 0   # Ensemble
PYTHONPATH=src python -m synib.entrypoints.train --config $B/dnr.json        --default_config $D --fold 0   # D&R
PYTHONPATH=src python -m synib.entrypoints.train --config $B/mmpareto.json   --default_config $D --fold 0   # MMPareto
PYTHONPATH=src python -m synib.entrypoints.train --config $B/reconboost.json --default_config $D --fold 0   # ReconBoost
PYTHONPATH=src python -m synib.entrypoints.train --config $B/mcr.json        --default_config $D --fold 0   # MCR
PYTHONPATH=src python -m synib.entrypoints.train --config $B/uni_text.json  --default_config $D --fold 0
PYTHONPATH=src python -m synib.entrypoints.train --config $B/uni_video.json --default_config $D --fold 0
# Vanilla fusion = the SynIB config with the penalty off:
PYTHONPATH=src python -m synib.entrypoints.train --config $B/synib.json --default_config $D --fold 0 --l 0
```

(For MUStARD use `run/configs/multibench/mustard/...`; for MOSI use
`run/configs/multibench/mosi/...`.)

---

## Hateful Memes (frozen CLIP-ViT + DeBERTa)

HM merges a **tier** (backbone) and a **method** config on top of the default config, so
use the wrapper (it does the merge). Tier = `small_tf_deberta`; run `--fold {0,1,2}`.

```bash
# SynIB — M_random
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json \
                  run/configs/hateful_memes/methods/synib_u.json \
                  --fold 0 --rmask random --l 0.01 --l_pareto 0.1 --perturb_pmin 0.3 --perturb_pmax 0.5

# SynIB — M_learned
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json \
                  run/configs/hateful_memes/methods/synib_u.json \
                  --fold 0 --rmask learned --l 0.01 --l_pareto 0.1 --perturb_lsparse 0.1

# baselines (swap the method config)
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/vanilla.json    --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/dnr.json        --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/mmpareto.json   --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/reconboost.json --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/mcr.json        --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/uni_text.json   --fold 0
./run/hateful_memes/train.sh run/configs/hateful_memes/tiers/small_tf_deberta.json run/configs/hateful_memes/methods/uni_image.json  --fold 0
```

Each wrapper call expands to `python -m synib.entrypoints.train --config <merged tier+method>
--default_config run/configs/hateful_memes/default_config_hm.json --fold k ...`. The paper
HM seeds are 109 / 27 / 3407.

---

## CREMA-D / CREMA-D-Irony (audio + visual)

Single self-contained config; the irony rate α is set with `--ironic_rate`
(Fig. 4 sweeps α ∈ {0.1, 0.3, 0.5, 0.8, 1.0, 2.0}). Run `--fold {0,1,2}`.

```bash
DEF=run/configs/cremad/default.json

# SynIB — M_random  (asymmetric variant: synib_u.json)
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/cremad/synib_u.json --default_config $DEF \
    --fold 0 --ironic_rate 1.0 --rmask random --l 1.0 --perturb_pmin 0.20 --perturb_fill ema

# SynIB — M_learned
PYTHONPATH=src python -m synib.entrypoints.train \
    --config run/configs/cremad/synib_u.json --default_config $DEF \
    --fold 0 --ironic_rate 1.0 --rmask learned --l 1.0 --perturb_lsparse 0.01 --perturb_fill ema

# baselines
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/ensemble.json        --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/dnr.json        --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/mmpareto.json   --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/reconboost.json --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/mcr.json        --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/uni_audio.json --default_config $DEF --fold 0 --ironic_rate 1.0
PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/cremad/uni_video.json --default_config $DEF --fold 0 --ironic_rate 1.0
```

The wrapper `./run/cremad/train.sh --rmask random --l 1.0 --pmin 0.20 --ironic_rate 1.0 --fold 0`
is equivalent for the SynIB run (it also exposes named scenarios, see `./run/cremad/train.sh --scenarios`).

---

## Synthetic XOR (mechanism figures)

These generate their own data in-code — no cache, just run them:

```bash
python scripts/analysis/xor_spurious_pub.py              # spurious-shortcut XOR (test acc vs β)
python scripts/analysis/Xor_PID3Main_MaskSynIB_Search.py # PID-controlled XOR simplex
python scripts/analysis/Xor_PIDExamineNTK.py             # per-source NTK diagnostics
python scripts/analysis/figs_pid_ntk_dynamics.py         # PID / NTK training-dynamics figure
```

---

## Common flags

| flag | meaning |
|---|---|
| `--fold {0,1,2}` | run / seed selector (report mean ± std over the three) |
| `--rmask {random,learned}` | mask construction: `M_random` or adversarial `M_learned` |
| `--l` | λ — weight on the SynIB KL penalty (`--l 0` = plain fusion) |
| `--l_pareto` | asymmetry ratio between the two counterfactual branches (`_U` objective) |
| `--perturb_pmin`, `--perturb_pmax` | Bernoulli mask probability range (random mask) |
| `--perturb_lsparse` | sparsity weight on the learned mask's inner loop |
| `--perturb_fill` | how masked features are filled (`ema`) |
| `--ironic_rate` | CREMA-D-Irony synergy density α |
| `--lr`, `--wd`, `--batch_size` | optimizer overrides |

Full list: `PYTHONPATH=src python -m synib.entrypoints.train --help`.
