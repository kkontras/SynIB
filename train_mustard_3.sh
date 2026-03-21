

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.001 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_text_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_video_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8



CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.00001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.001 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --validate_with accuracy --batch_size 8

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_text_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_text_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_text_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_video_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_video_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.001 --validate_with accuracy --batch_size 8
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_video_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8


CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synibu_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8 --l 1 --perturb learned --perturb_fill ema --perturb_lsparse 0.1




CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 4 --l 1 --perturb learned --perturb_fill ema --perturb_lsparse 0.1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 4 --l 1 --perturb learned --perturb_fill ema --perturb_lsparse 1
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 4 --l 1 --perturb learned --perturb_fill ema --perturb_lsparse 10


CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.1 --perturb learned --perturb_fill ema --perturb_lsparse 0.1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.1 --perturb learned --perturb_fill ema --perturb_lsparse 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.1 --perturb learned --perturb_fill ema --perturb_lsparse 10

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 10 --perturb learned --perturb_fill ema --perturb_lsparse 0.1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 10 --perturb learned --perturb_fill ema --perturb_lsparse 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 10 --perturb learned --perturb_fill ema --perturb_lsparse 10

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.01 --perturb learned --perturb_fill ema --perturb_lsparse 0.1
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.01 --perturb learned --perturb_fill ema --perturb_lsparse 1
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.01 --perturb learned --perturb_fill ema --perturb_lsparse 10

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.001 --perturb learned --perturb_fill ema --perturb_lsparse 0.1
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.001 --perturb learned --perturb_fill ema --perturb_lsparse 1
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_synib_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 4 --l 0.001 --perturb learned --perturb_fill ema --perturb_lsparse 10



CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_iha.json  --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8 --iha_lr 0.0005 --pseudo_heads_q 16 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_iha_lora.json  --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8 --iha_lr 0.0005 --pseudo_heads_q 16 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_full_ft.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.00005 --wd 0.01 --validate_with accuracy --batch_size 8                                                                         
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m synib.entrypoints.train --config run/configs/MUStARD/cache_full_ft.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.00005 --wd 0.01 --validate_with accuracy --batch_size 8 --finetune_layers "20,21,22,23,24,25,26,27"    


CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_iha.json  --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.01 --validate_with accuracy --batch_size 8 --iha_lr 0.0005 --pseudo_heads_q 16 --pseudo_heads_kv 8 --iha_init identity --iha_layers all
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python -m synib.entrypoints.show --config run/configs/MUStARD/cache_lora.json --default_config run/configs/MUStARD/default_config_mustard_cache_mib.json --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy --batch_size 8
