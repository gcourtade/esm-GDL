
# esm-GDL
This is a fork of https://github.com/cicese-biocom/esm-AxP-GDL/


## Installation

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

## Run
train.sh modified to use pre-calculated structures