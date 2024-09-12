#!/bin/bash
#SBATCH --job-name=DPO_Pref
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --output=logs/%j.log

source /data/will_cai/miniconda3/etc/profile.d/conda.sh

conda activate /data/will_cai/wicai24/ENTER/envs/repe_env

cd /data/will_cai/wicai24/RepE-pref-learning/code

bash run.sh