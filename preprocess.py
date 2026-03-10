import os
import numpy as np
import pandas as pd
import torch
import logging
import random
import pickle
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch_geometric.data as geo_data
from sklearn.model_selection import StratifiedKFold
# from torch_geometric.utils import smiles_to_graph

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path):

    logger.info(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
      
        if not all(col in data.columns for col in ["Products", "EC", "Reactiants"]):
            raise ValueError(f"File {file_path} missing required columns ('Products', 'EC' or 'Reactiants')")
        data = data.dropna(subset=["Products", "EC", "Reactiants"])
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        raise


def create_label_mapping(train_data):
 
    train_labels = train_data["EC"].values.tolist()
    unique_labels = sorted(list(set(train_labels)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    logger.info(f"Created label mapping with {len(unique_labels)} unique classes")
    return label_to_id, id_to_label


def filter_unknown_labels(data, label_to_id):
  
    original_count = len(data)
    data = data[data["EC"].isin(label_to_id.keys())]
    filtered_count = original_count - len(data)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} samples with unknown labels")
    return data


def smiles_to_mol_graph(smiles):
  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 获取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalValence(),
            atom.GetHybridization().real,
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing())
        ]
        atom_features.append(features)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  
    
    atom_features = np.array(atom_features, dtype=np.float32)
    edge_index = np.array(edge_index, dtype=np.int64).T
    
    return {
        "x": torch.tensor(atom_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long)
    }

def tokenize_data(tokenizer, reactiants_smiles_list, products_smiles_list, labels, max_length=512):
    try:
    
        reactiants_encodings = tokenizer(
            reactiants_smiles_list,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
       
        products_encodings = tokenizer(
            products_smiles_list,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "reactiants_input_ids": reactiants_encodings["input_ids"],
            "reactiants_attention_mask": reactiants_encodings["attention_mask"],
            "products_input_ids": products_encodings["input_ids"],
            "products_attention_mask": products_encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise

def process_graph_data(smiles_list):

    graph_data_list = []
    for smiles in smiles_list:
        graph = smiles_to_mol_graph(smiles)
        if graph is not None:
            graph_data_list.append(graph)
        else:           
            graph_data_list.append(None)
    return graph_data_list


def main():
    
    train_data = load_data("train.csv")
    valid_data = load_data("valid.csv")
    test_data = load_data("test.csv")
    
 
    all_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
    logger.info(f"Merged all data into {len(all_data)} samples")


    label_to_id, id_to_label = create_label_mapping(all_data)


    all_data = filter_unknown_labels(all_data, label_to_id)

   
    all_products_smiles = all_data["Products"].values.tolist()
    all_reactiants_smiles = all_data["Reactiants"].values.tolist()
    all_labels = [label_to_id[label] for label in all_data["EC"].values.tolist()]


    combined = list(zip(all_reactiants_smiles, all_products_smiles, all_labels))
    random.shuffle(combined)
    all_reactiants_smiles, all_products_smiles, all_labels = zip(*combined)
    all_reactiants_smiles = list(all_reactiants_smiles) 
    all_products_smiles = list(all_products_smiles)  
    all_labels = list(all_labels)  

    
    local_model_path = "zinc250k_v2_40k"
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise

   
    all_smiles_combined = all_reactiants_smiles + all_products_smiles
    max_len = max(len(tokenizer.encode(smiles, add_special_tokens=True)) for smiles in all_smiles_combined)
    max_seq_length = min(512, max(300, max_len))
    logger.info(f"Max sequence length: {max_seq_length} (based on all data)")

    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
   
    logger.info("Processing graph data for all samples...")
    all_graphs = process_graph_data(all_products_smiles)
    
   
    logger.info("Filtering invalid graph data...")
    valid_indices = [i for i, graph in enumerate(all_graphs) if graph is not None]
    filtered_reactiants_smiles = [all_reactiants_smiles[i] for i in valid_indices]
    filtered_products_smiles = [all_products_smiles[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]
    filtered_graphs = [all_graphs[i] for i in valid_indices]
    logger.info(f"Filtered out {len(all_products_smiles) - len(filtered_products_smiles)} samples with invalid graph data")
    
  
    logger.info("Tokenizing data...")
    all_encodings = tokenize_data(tokenizer, filtered_reactiants_smiles, filtered_products_smiles, filtered_labels, max_seq_length)

    
    cache_dir = "preprocessed"
    os.makedirs(cache_dir, exist_ok=True)

   
    for fold, (train_idx, test_idx) in enumerate(skf.split(filtered_products_smiles, filtered_labels)):
        logger.info(f"Processing fold {fold+1}/5...")
                
       
        train_encodings_fold = {
            k: v[train_idx] for k, v in all_encodings.items()
        }
        test_encodings_fold = {
            k: v[test_idx] for k, v in all_encodings.items()
        }
        
       
        train_graphs_fold = [filtered_graphs[i] for i in train_idx]
        test_graphs_fold = [filtered_graphs[i] for i in test_idx]
        
        
        fold_dir = os.path.join(cache_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
     
        torch.save(train_encodings_fold, os.path.join(fold_dir, "train_encodings.pt"))
        torch.save(test_encodings_fold, os.path.join(fold_dir, "test_encodings.pt"))
        
      
        torch.save(train_graphs_fold, os.path.join(fold_dir, "train_graphs.pt"))
        torch.save(test_graphs_fold, os.path.join(fold_dir, "test_graphs.pt"))
        
        logger.info(f"Fold {fold+1} saved - Train: {len(train_idx)}, Test: {len(test_idx)}")

   
    with open(os.path.join(cache_dir, "label_map.pkl"), "wb") as f:
        pickle.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f)

    
    with open(os.path.join(cache_dir, "label_map.txt"), "w") as f:
        for label, idx in label_to_id.items():
            f.write(f"{label}\t{idx}\n")

    logger.info("Preprocessing completed successfully")
    logger.info(f"Final dataset size after filtering: {len(filtered_products_smiles)}")
    logger.info(f"5-fold cross-validation data saved to {cache_dir}")


if __name__ == "__main__":
    main()