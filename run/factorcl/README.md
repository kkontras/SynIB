# FactorCL Affect Datasets

This directory exposes the FactorCL affect datasets in SynIB-native form.

Supported targets:
- `mosi-vt`
- `mosi-vta`
- `mosei-vt`
- `urfunny-vt`
- `mustard-vt`

## Download prepared data

Prepared pickles are downloaded into `src/synib/mydatasets/Factor_CL_Datasets/prepared/` by default.

```bash
./run/factorcl/download.sh all
```

Per dataset:

```bash
./run/factorcl/download.sh mosi
./run/factorcl/download.sh mosei
./run/factorcl/download.sh urfunny
./run/factorcl/download.sh mustard
```

You can override the root with `SYNIB_FACTORCL_DATA_ROOT=/path/to/data`.

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
