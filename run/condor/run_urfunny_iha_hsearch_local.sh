#!/usr/bin/env bash
set -euo pipefail

cd /esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB

# Complete partially finished 3-fold settings
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 2 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 4 --pseudo_heads_kv 4 --iha_init identity --iha_layers 2,3
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 2 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 4 --pseudo_heads_kv 4 --iha_init identity --iha_layers all --iha_lr 0.0005
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 0 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 4 --pseudo_heads_kv 4 --iha_init identity_noise --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 0 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 4 --pseudo_heads_kv 4 --iha_init orthogonal --iha_layers all

# Head-count variants with no checkpoints yet
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 0 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 4 --iha_init identity --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 1 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 4 --iha_init identity --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 2 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 4 --iha_init identity --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 0 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 1 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
./run/condor/run_train.sh --config run/configs/FactorCL/URFunny/release/VT/joint_tf_iha.json --default_config run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json --fold 2 --validate_with accuracy --batch_size 32 --no_model_save --lr 0.001 --wd 0.0001 --pseudo_heads_q 8 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
