[![Made with Python](https://img.shields.io/badge/Python-3.7-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/docs/1.12/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3.1-%237732a8.svg?style=flat&logo=PyG&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/2.3.1/)
[![CUDA](https://img.shields.io/badge/CUDA-11-%2376B900.svg?style=flat&logo=NVIDIA&logoColor=white)](https://developer.nvidia.com/cuda-11-3-1-download-archive)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

# **esm-AxP-GDL**

esm-AxP-GDL is a framework to build Graph Deep Learning (GDL)-based models leveraging ESMFold-predicted peptide 
structures and ESM-2 based amino acid-level characteristics for the prediction of antimicrobial peptides (AMPs). 
This framework was designed to be easily extended to modeling any task related to the prediction of peptide and 
protein biological activities (or properties).

![workflow_framework](https://github.com/cicese-biocom/esm-AxP-GDL/assets/136017848/99191e5d-d1a5-470b-a905-126bf96e307f)

## **Install esm-AxP-GDL**
Ensure you have nvcc installed

Clone the repository:
```
git clone https://github.com/cicese-biocom/esm-AxP-GDL.git
```

Create environment:
```
mamba create -n esm-gdl python=3.9
```

Install pytorch and torch_geometric, for us:
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

```

Install esmfold and its dependencies dlogger and openfold
```
pip install "fair-esm[esmfold]"

# Not important if we are not running esmfold
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
```

**Not important if we are not running esmfold**
Install openfold (esmfold dependency) 
```
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
git checkout tags/v1.0.1
sed -i 's/c++14/c++17/g' setup.py
pip install .
```

Install remaining requirements
```
pip install -r requirements.txt
```




NOTE: we provide template scripts to run training/test/inference Slurm batch jobs.

## **Usage**
### **Input data format**
The framework esm-AxP-GDL is inputted with a comma separated value (CSV) file, which contains 
the identifier, the amino acid sequence, the activity value, and the partition of each peptide. 
We used the numbers 1, 2 and 3 to represent the training, validation, and test sets, respectively. 
For training or using a model for inference, it should be specified the path for the input CSV file.

### **For training or using a model for inference**
train.py and test.py are used to carry out the training and inference steps, respectively. 
The next command lines can be used to run the training and inference steps, respectively.

#### Train
```
usage: train.py [-h] --dataset DATASET [--tertiary_structure_method {esmfold}]
                [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
                --gdl_model_path GDL_MODEL_PATH
                [--esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t33,esm2_t36,esm2_t48}]
                [--edge_construction_functions EDGE_CONSTRUCTION_FUNCTIONS]
                [--distance_function {euclidean,canberra,lance_williams,clark,soergel,bhattacharyya,angular_separation}]
                [--distance_threshold DISTANCE_THRESHOLD]
                [--amino_acid_representation {CA}]
                [--number_of_heads NUMBER_OF_HEADS]
                [--hidden_layer_dimension HIDDEN_LAYER_DIMENSION]
                [--add_self_loops] [--use_edge_attr]
                [--learning_rate LEARNING_RATE] [--dropout_rate DROPOUT_RATE]
                [--number_of_epochs NUMBER_OF_EPOCHS] [--save_ckpt_per_epoch]
                [--validation_mode {random_coordinates,random_embeddings}]
                [--randomness_percentage RANDOMNESS_PERCENTAGE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to save/load the models
  --esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t33,esm2_t36,esm2_t48}
                        ESM-2 model to be used
  --edge_construction_functions EDGE_CONSTRUCTION_FUNCTIONS
                        Criteria (e.g., distance) to define a relationship
                        (graph edges) between amino acids. Only one ESM-2
                        contact map can be specified. The options available
                        are: 'distance_based_threshold', 'sequence_based',
                        'esm2_contact_map_50', 'esm2_contact_map_60',
                        'esm2_contact_map_70', 'esm2_contact_map_80',
                        'esm2_contact_map_90'
  --distance_function {euclidean,canberra,lance_williams,clark,soergel,bhattacharyya,angular_separation}
                        Distance function to construct graph edges
  --distance_threshold DISTANCE_THRESHOLD
                        Distance threshold to construct graph edges
  --amino_acid_representation {CA}
                        Reference atom into an amino acid to define a
                        relationship (e.g., distance) regarding another amino
                        acid
  --number_of_heads NUMBER_OF_HEADS
                        Number of heads
  --hidden_layer_dimension HIDDEN_LAYER_DIMENSION
                        Hidden layer dimension
  --add_self_loops      True if specified, otherwise, False. True indicates to
                        use auto loops in attention layer.
  --use_edge_attr       True if specified, otherwise, False. True indicates to
                        use edge attributes in graph learning.
  --learning_rate LEARNING_RATE
                        Learning rate
  --dropout_rate DROPOUT_RATE
                        Dropout rate
  --number_of_epochs NUMBER_OF_EPOCHS
                        Maximum number of epochs
  --save_ckpt_per_epoch
                        True if specified, otherwise, False. True indicates
                        that the models of every epoch will be saved. False
                        indicates that the latest model and the best model
                        regarding the MCC metric will be saved.
  --validation_mode {random_coordinates,random_embeddings}
                        Criteria to corroborate that the predictions of the
                        models are not by chance
  --randomness_percentage RANDOMNESS_PERCENTAGE
                        Percentage of rows to be randomly generated     
```

#### Test
```
usage: test.py [-h] --dataset DATASET [--tertiary_structure_method {esmfold}]
               [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
               --gdl_model_path GDL_MODEL_PATH [--dropout_rate DROPOUT_RATE]
               --output_path OUTPUT_PATH [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to load the model
  --output_path OUTPUT_PATH
                        The path where the output data will be saved.
  --seed SEED           Seed to run the test                                          
```
#### Inference
```
usage: inference.py [-h] --dataset DATASET
                    [--tertiary_structure_method {esmfold}]
                    [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
                    --gdl_model_path GDL_MODEL_PATH
                    [--dropout_rate DROPOUT_RATE] --output_path OUTPUT_PATH
                    [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to load the model
  --output_path OUTPUT_PATH
                        The path where the output data will be saved.
  --seed SEED           Seed to run the Inference
                                                
```

### **Example**
We provide the train.sh and test.sh example scripts to train or use a model for inference, respectively.
In these scripts are used the AMPDiscover dataset as input set, the model `esm2_t36_3B_UR50D` to evolutionary 
characterize the graph nodes, a `distance threshold equal to 10 angstroms`
to build the graph edges, and a `hidden layer size equal to 128`.

When using the Docker container, the example scripts should be used as follows:
```
docker-compose run --rm esm-axp-gdl-env sh train.sh
```
```
docker-compose run --rm esm-axp-gdl-env sh test.sh
```
```
docker-compose run --rm esm-axp-gdl-env sh inference.sh
```

### **Best models**
Best models created. So far, the only models are to predict general-AMP.  

| Name                                                                                                                      | Dataset                                                          | Endpoint     | MCC    | Description                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|--------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [amp_esmt33_d10_hd128_(Model2).pt](https://drive.google.com/file/d/1mskGXsYz5yjNxQUoJwRWDHi_it1bORoG/view?usp=sharing)    | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9389 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t33_650M_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 128`. |
| [amp_esmt36_d10_hd128_(Model3).pt](https://drive.google.com/file/d/1pBkNn6-_6w5YO2TljMkOVo5xVivnAQAf/view?usp=sharing)    | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9505 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t36_3B_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 128`.   |
| [amp_esmt30_d15_hd128_(Model5).pt](https://drive.google.com/file/d/1gvGDVTCQ0QmTP6rU9tSBC9rc1e4BV-M-/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9379 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t30_150M_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 15 angstroms` to build the graph edges, and a `hidden layer size equal to 128`. |

NOTE:  The performance `metrics` obtained and `parameters` used to build the best models are available at `/best_models` directory. The models are available-freely making click on the Table.