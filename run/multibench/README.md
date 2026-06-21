# MultiBench affective datasets

The three MultiBench affective benchmarks (CMU-MOSI, UR-Funny, MUStARD) in
SynIB-native form, using frozen text+visual (V+T) features.

Targets: `mosi-vt`, `urfunny-vt`, `mustard-vt`.

## Download raw data and build the cache

```bash
./run/multibench/download_and_build_cache.sh all          # mosi + urfunny + mustard
./run/multibench/download_and_build_cache.sh mosi         # or: urfunny | mustard
./run/multibench/download_and_build_cache.sh mosi --skip-download --cache-root /data/my_cache
```

For local data sources, set the per-dataset env vars listed in the script's `--help`.

## Train

```bash
# SynIB (random / learned mask) — MOSI uses the asymmetric variant (synib_u)
./run/multibench/train.sh mosi-vt    run/configs/multibench/mosi/synib_u.json --fold 0 --rmask random --l 0.1
./run/multibench/train.sh urfunny-vt run/configs/multibench/urfunny/synib.json --fold 0 --rmask learned --l 1

# baselines
./run/multibench/train.sh mustard-vt run/configs/multibench/mustard/reconboost.json --fold 0
./run/multibench/train.sh urfunny-vt run/configs/multibench/urfunny/dnr.json --fold 0
```

Each dataset directory (`run/configs/multibench/{mosi,urfunny,mustard}/`) holds a
`default.json` (dataset/optimizer base) plus one config per method: `synib.json`,
`synib_u.json`, the baselines (`dnr`, `mcr`, `mmpareto`, `reconboost`, `ensemble`), and
the unimodal references (`uni_text.json`, `uni_video.json`). The full per-cell commands
are in [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md).

## Show

```bash
./run/multibench/show.sh mustard-vt run/configs/multibench/mustard/synib.json --fold 0
```
