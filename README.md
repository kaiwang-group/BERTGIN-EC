# Multi-modal Enzyme Classification Prediction Model

## Overview

```BERTGIN-EC``` is a multi-modal fusion method based on chemical language models and graph neural networks, designed to predict EC number classification of enzyme-catalyzed reactions. This method simultaneously utilizes SMILES sequence information of reactants and products as well as molecular graph structure information, achieving efficient enzyme classification prediction through various fusion strategies.

## Description

This project proposes an innovative multi-modal fusion framework that combines the ChemBERTa chemical language model and graph neural networks to predict EC numbers of enzyme-catalyzed reactions. The method leverages SMILES sequence information of reactants and products along with molecular graph structure information, achieving efficient enzyme classification prediction through various fusion strategies.

The project includes the following core files and directory structure:
- `multimodal_model.py` — Multi-modal fusion model definition
- `train.py` — Model training script
- `test.py` — Model testing script
- `preprocess.py` — Data preprocessing script
- `ablation_model.py` — Ablation study model definition
- `train_ablation.py` — Ablation study script
- `data/` — Reaction data directory



## System Requirements

This project is developed based on the Python deep learning framework and requires the following environment dependencies:

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


