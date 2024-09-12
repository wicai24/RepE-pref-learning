#!/bin/bash
#SBATCH --job-name=DPO_Pref
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --output=logs/%j.log

source /data/will_cai/miniconda3/etc/profile.d/conda.sh

conda activate /data/will_cai/wicai24/ENTER/envs/repe_env

cd /data/will_cai/wicai24/RepE-pref-learning/code

python prefix_tuning.py \
    --model_path "../model/SFT/final_checkpoint" \
    --data_path "../data/ultrafeedback/rm" \
    --output_dir "../model/DPO/input_tuning_prefix_MSE_retain" \
    --use_neg True \
    --use_retain True \
    --use_cosine False