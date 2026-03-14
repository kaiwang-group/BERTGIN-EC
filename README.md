# Multi-modal Enzyme Classification Prediction Model

## Overview

```BERTGIN-EC``` is a multi-modal fusion method based on chemical language models and graph neural networks, designed to predict EC number classification of enzyme-catalyzed reactions. This method simultaneously utilizes SMILES sequence information of reactants and products as well as molecular graph structure information, achieving efficient enzyme classification prediction through various fusion strategies.

```BERTGIN-EC``` consists of three modules: the reaction SMILES sequence feature extraction module,  reaction graph feature extraction module,  feature fusion and prediction module.The reaction SMILES sequence feature extraction module employs a pre-trained BERT model to extract the 1D sequence features of reaction SMILES.The reaction graph feature extraction module uses a Graph Isomorphism Network (GIN) to extract the 2D graph structural features of the reaction.The feature fusion and prediction module fuses the bimodal features to predict and output the EC number.

The overall framework of ```BERTGIN-EC``` is shown in the following figure.

![BERTGIN-EC Model Architecture](https://github.com/huajiqing23/BERTGIN-EC/blob/main/Model%20Architecture.png)


## Description

The project includes the following core files and directory structure:
- The folder `data/` contains the directory for storing the reaction data utilized in the training, testing and ablation study of BERTGIN-EC.
- The file `ablation_model.py` contains the definition of the ablation study model for BERTGIN-EC.
- The file `multimodal_model.py` contains the definition of the multi-modal fusion model for BERTGIN-EC.
- The file `preprocess.py` contains the script for preprocessing the reaction data used in BERTGIN-EC.
- The file `test.py` contains the script for training the BERTGIN-EC model.
- The file `train.py` contains the script for testing the BERTGIN-EC model.
- The file `train_ablation.py` contains the script for conducting the ablation study of BERTGIN-EC.

The datasets after 5-fold cross-validation processing and the pre-trained models can be accessed via the following link: https://pan.baidu.com/s/1AeTamOZOdDfgRH7D_V6vNA?pwd=3us4.

## System Requirements

The proposed ```BERTGIN-EC``` has been implemented, trained, and tested by using `Python 3.8` and `PyTorch 2.1.0` with `CUDA 12.1` and an `NVIDIA RTX4090` graphics card.

The package depends on the Python scientific stack:
```
Python
torch
Transformers
PyTorch Geometric
RDKit
scikit-learn
pandas
numpy
tqdm
networkx
scipy
```

Dependencies can be installed using the following command:

```bash
pip install torch transformers torch-geometric rdkit scikit-learn pandas numpy tqdm networkx scipy
```

## Usage

### Data Preprocessing

```bash
python preprocess.py
```

This script will perform the following operations: load the dataset; convert SMILES to molecular graph structures; encode SMILES using the ChemBERTa tokenizer; perform 5-fold cross-validation data splitting; save preprocessed data to the `preprocessed/` directory.

### Model Training

```bash
python train.py
```

Training parameters can be modified in `train.py`:

```python
params = {
    'Epoch': 40,              # Number of training epochs
    'batch_size': 64,         # Batch size
    'local_model_path': "/path/to/chemberta",  # ChemBERTa model path
    'fusion_type': "attention",  # Fusion type: concat/sum/attention
    'head_type': "mlp",       # Classification head type: simple/mlp
    'gnn_type': "gin"         # GNN type: gcn/gat/gin
}
```
Modify GNN Architecture

Modify the `GraphFeatureExtractor` parameters in `multimodal_model.py`:

```python
self.graph_extractor = GraphFeatureExtractor(
    in_channels=8,          # Input feature dimension
    hidden_channels=        # Hidden layer dimension
    out_channels=           # Output feature dimension
    num_layers=3,           # Number of GNN layers
    gnn_type="gin"          # GNN type: gcn/gat/gin
)
```


### Model Testing

```bash
python test.py
```

### Ablation Study

```bash
python train_ablation.py
```

The ablation study supports evaluating the contributions of different modalities: graph-only modality, sequence-only modality, and the combination of sequence and graph modalities.


## Pre-trained Models

This project uses ChemBERTa as the sequence feature extractor. Pre-trained weights can be downloaded from Hugging Face:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
```


