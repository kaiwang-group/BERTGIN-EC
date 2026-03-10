# Multi-modal Enzyme Classification Prediction Model

A multi-modal fusion approach based on chemical language models and graph neural networks for enzyme catalytic reaction classification (EC number) prediction.

## Project Overview

This project proposes an innovative multi-modal fusion framework that combines chemical language models (ChemBERTa) and graph neural networks (GNN) to predict enzyme catalytic reaction EC numbers. The method simultaneously utilizes SMILES sequence information of reactants and products as well as molecular graph structure information, achieving efficient enzyme classification prediction through various fusion strategies.

## Project Structure

```
multi_ec/
├── multimodal_model.py      # Multi-modal fusion model definition
├── train.py    			 # Model training script
├── test.py       			 # Model testing script
├── preprocess.py            # Data preprocessing script
├── ablation_model.py        # Ablation study model definition
├── train_ablation.py        # Ablation study script
├── preprocessed2/           # Preprocessed data directory
│   ├── fold_1/ ~ fold_5/    # 5-fold cross-validation data
│   ├── label_map.pkl        # Label mapping file
│   └── *.pt                 # Encoding and graph data files
└── save_multimodal_fold*_*/ # Trained model save directory
```

## Environment Requirements

- Python 
- PyTorch
- Transformers 
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas
- numpy

### Install Dependencies

```bash
pip install torch transformers torch-geometric rdkit scikit-learn pandas numpy
```

## Usage

### 1. Data Preprocessing

```bash
python preprocess.py
```

This script will:
- Load training, validation, and test data
- Convert SMILES to molecular graph structures
- Encode SMILES using ChemBERTa tokenizer
- Perform 5-fold cross-validation data splitting
- Save preprocessed data to `preprocessed2/` directory

### 2. Model Training

```bash
python train_multimodal.py
```

Training parameters (modify in `train_multimodal.py`):

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

### 3. Model Testing

```bash
python test_multimodal.py
```

### 4. Ablation Study

```bash
python ablation_study.py
```

Ablation study supports evaluating contributions from different modalities:
- Graph-only modality
- Sequence-only modality
- Sequence and  Graph modality

## Model Architecture

### Multi-modal Fusion Model

```
┌─────────────────────────────────────────────────────────────┐
│                    MultimodalFusionModel                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │   Reactiants     │      │    Products      │            │
│  │   SMILES         │      │    SMILES        │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                       │
│           ▼                         ▼                       │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │   ChemBERTa      │      │   ChemBERTa      │            │
│  │   (Sequence)     │      │   (Sequence)     │            │
│  └────────┬─────────┘      └────────┬─────────┘            │
│           │                         │                       │
│           └───────────┬─────────────┘                       │
│                       │                                     │
│                       ▼                                     │
│              ┌──────────────────┐                          │
│              │  Concatenation   │                          │
│              └────────┬─────────┘                          │
│                       │                                     │
│  ┌──────────────────┐ │ ┌──────────────────┐              │
│  │   Products       │ │ │   Fusion Layer   │              │
│  │   Graph          │─┘ │   (Concat/Sum/   │              │
│  │   (GNN)          │   │   Attention)     │              │
│  └──────────────────┘   └────────┬─────────┘              │
│                                  │                         │
│                                  ▼                         │
│                         ┌──────────────────┐              │
│                         │  Classification  │              │
│                         │  Head (MLP)      │              │
│                         └────────┬─────────┘              │
│                                  │                         │
│                                  ▼                         │
│                         ┌──────────────────┐              │
│                         │  EC Prediction   │              │
│                         └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy**: Classification accuracy
- **F1 Score**: F1 score (weighted average)
- **Precision**: Precision (weighted average)
- **Recall**: Recall (weighted average)
- **MCC**: Matthews Correlation Coefficient

## Pre-trained Models

This project uses [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry) as the sequence feature extractor. Pre-trained weights can be downloaded from Hugging Face:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
```

## Core Code Explanation

### 1. Molecular Graph Feature Extraction

```python
class GraphFeatureExtractor(nn.Module):
    """Graph Neural Network for extracting graph-level features from molecular graphs."""
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=256, 
                 num_layers=3, gnn_type="gcn"):
        # Supports three architectures: GCN/GAT/GIN
        ...
```

### 2. Multi-modal Fusion

```python
class MultimodalFusionModel(nn.Module):
    """Multimodal model combining ChemBERTa sequence features with GNN graph features."""
    def __init__(self, model_path, num_labels, fusion_type="concat", 
                 head_type="mlp", gnn_type="gcn"):
        # Fusion type: concat/sum/attention
        # Classification head type: simple/mlp
        # GNN type: gcn/gat/gin
        ...
```


## Custom Configuration

### Modify GNN Architecture

Modify the parameters of `GraphFeatureExtractor` in `multimodal_model.py`:

```python
self.graph_extractor = GraphFeatureExtractor(
    in_channels=8,          # Input feature dimension
    hidden_channels=128,    # Hidden layer dimension
    out_channels=256,       # Output feature dimension
    num_layers=3,           # Number of GNN layers
    gnn_type="gin"          # GNN type: gcn/gat/gin
)
```

### Modify Fusion Strategy

Modify the fusion type in the training script:

```python
params = {
    'fusion_type': "attention",  # Options: "concat", "sum", "attention"
    'head_type': "mlp",          # Options: "simple", "mlp"
    'gnn_type': "gin"            # Options: "gcn", "gat", "gin"
}
```

## Notes

1. **Data Paths**: Please ensure data path configurations are modified before running scripts
2. **GPU Memory**: Large batch sizes may require significant GPU memory, adjust according to your hardware
3. **RDKit**: Ensure RDKit is properly installed for SMILES conversion
4. **PyTorch Geometric**: Installation must match PyTorch version
