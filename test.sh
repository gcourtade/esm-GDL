#!/bin/bash
#
#SBATCH --job-name=esm-GDL-test
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1


# We used the numbers 1, 2 and 3 to represent the training, validation, and test sets, respectively. 
dataset="cbms/Avicel_data.csv"
pdb_path="cbms/ESMFold_pdbs/"
tertiary_structure_method="esmfold"
gdl_model_path="cbms/gdl_best_model_so_far.pt"
output_path="cbms/output/"
batch_size=256


python=/triumvirate/apps/miniforge3/envs/esm-gdl/bin/python


$python test.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path" \
    --output_path="$output_path"  \
    --batch_size="$batch_size"

