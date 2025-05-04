#!/bin/bash

dataset="cbms/Avicel_data.csv"
pdb_path="cbms/ESMFold_pdbs/"
tertiary_structure_method="esmfold"
gdl_model_path="cbms/output/"
esm2_representation="esm2_t33"
edge_construction_functions="distance_based_threshold"
distance_function="euclidean"
distance_threshold=10
amino_acid_representation="CA"
number_of_heads=8
hidden_layer_dimension=16
learning_rate=0.0001
dropout_rate=0.1
batch_size=512
number_of_epoch=200

#rm -r $pdb_path
#mkdir -p $pdb_path
rm -r $gdl_model_path

python=/triumvirate/apps/miniforge3/envs/esm-gdl/bin/python
$python train.py \
    --dataset "$dataset" \
    --pdb_path "$pdb_path" \
    --gdl_model_path="$gdl_model_path"  \
    --esm2_representation "$esm2_representation" \
    --edge_construction_functions="$edge_construction_functions" \
    --distance_function="$distance_function" \
    --distance_threshold="$distance_threshold" \
    --amino_acid_representation="$amino_acid_representation" \
    --number_of_heads="$number_of_heads" \
    --hidden_layer_dimension="$hidden_layer_dimension" \
    --add_self_loops \
    --use_edge_attr \
    --learning_rate="$learning_rate" \
    --dropout_rate="$dropout_rate" \
    --batch_size="$batch_size" \
    --number_of_epoch="$number_of_epoch" \
    --save_ckpt_per_epoch
#    --tertiary_structure_method="$tertiary_structure_method"  \ # None means use existing
