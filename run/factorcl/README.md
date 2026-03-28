# FactorCL Affect Datasets

This directory exposes the FactorCL affect datasets in SynIB-native form.

Supported targets:
- `mosi-vt`
- `mosi-vta`
- `mosei-vt`
- `urfunny-vt`
- `mustard-vt`

## Download Raw Data And Build Cache

Raw roots are materialized under `src/synib/mydatasets/Factor_CL_Datasets/raw/` by default and caches under `artifacts/cache/factorcl/`.

```bash
./run/factorcl/download_and_build_cache.sh all
```

Per dataset:

```bash
./run/factorcl/download_and_build_cache.sh mosi
./run/factorcl/download_and_build_cache.sh mosei
./run/factorcl/download_and_build_cache.sh urfunny
./run/factorcl/download_and_build_cache.sh mustard
```

Useful options:

```bash
./run/factorcl/download_and_build_cache.sh all --gpu 0 --cache-root /data/factorcl_cache
./run/factorcl/download_and_build_cache.sh mosi --skip-download
```

The combined script expects canonical raw metadata for `MOSI`, `MOSEI`, and `URFunny`. For local sources, set the corresponding environment variables listed in the script help.

## Train

```bash
./run/factorcl/train.sh mosi-vt run/configs/FactorCL/Mosi/release/VT/unimodal_text.json --fold 0
./run/factorcl/train.sh mosi-vta run/configs/FactorCL/Mosi/release/VTA/unimodal_audio.json --fold 0
./run/factorcl/train.sh mosei-vt run/configs/FactorCL/Mosei/syn/VT/synprom_RMask.json --fold 0 --rmask random --l 1.0 --pmin 0.2
./run/factorcl/train.sh urfunny-vt run/configs/FactorCL/URFunny/release/VT/ReconBoost.json --fold 0
./run/factorcl/train.sh mustard-vt run/configs/FactorCL/Mustard/release/VT/DnR.json --fold 0
```

## Show

```bash
./run/factorcl/show.sh mosi-vt run/configs/FactorCL/Mosi/release/VT/MCR.json --fold 0
```
