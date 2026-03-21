#!/bin/bash
# Submits each training line in the synib scripts as a separate SLURM job.
# Usage:           bash submit_synib.sh
# Dry run preview: DRY_RUN=1 bash submit_synib.sh

SLURM_OPTS="--gres=gpu:1 --time=60:00:00 -p pi_ppliang --nodelist=node2500 -c 15 --mem=80G"
WORKDIR=$(pwd)
CONDA_INIT="source ~/miniconda3/bin/activate synergy_new"

for script in train_mustard_synib_lora.sh train_mustard_synibu_lora.sh; do
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        # SLURM assigns the GPU; fix device index to 0
        cmd=$(echo "$line" | sed 's/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=0/')
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "sbatch $SLURM_OPTS --wrap=\"cd $WORKDIR && $CONDA_INIT && $cmd\""
        else
            sbatch $SLURM_OPTS --wrap="cd $WORKDIR && $CONDA_INIT && $cmd"
        fi
    done < "$script"
done
