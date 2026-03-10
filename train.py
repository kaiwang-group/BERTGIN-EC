import os
import torch
import logging
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader
from multimodal_model import load_multimodal_model
import sklearn
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalSmilesDataset(Dataset):
    def __init__(self, encodings, graphs):
        self.encodings = encodings
        self.graphs = graphs
        assert len(encodings["labels"]) == len(graphs), "Encodings and graphs must have the same length"

    def __getitem__(self, idx):
        item = {
            "reactiants_input_ids": self.encodings["reactiants_input_ids"][idx],
            "reactiants_attention_mask": self.encodings["reactiants_attention_mask"][idx],
            "products_input_ids": self.encodings["products_input_ids"][idx],
            "products_attention_mask": self.encodings["products_attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
            "x": self.graphs[idx]["x"],
            "edge_index": self.graphs[idx]["edge_index"]
        }
        return item

    def __len__(self):
        return len(self.encodings["labels"])


def multimodal_collate_fn(batch):

    reactiants_input_ids = torch.stack([item["reactiants_input_ids"] for item in batch])
    reactiants_attention_mask = torch.stack([item["reactiants_attention_mask"] for item in batch])
    products_input_ids = torch.stack([item["products_input_ids"] for item in batch])
    products_attention_mask = torch.stack([item["products_attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    
    xs = [item["x"] for item in batch]
    edge_indices = [item["edge_index"] for item in batch]
    
    
    batch_indices = []
    for i, x in enumerate(xs):
        batch_indices.append(torch.full((x.size(0),), i, dtype=torch.long))
    
    
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat([ei + sum(x.size(0) for x in xs[:i]) for i, ei in enumerate(edge_indices)], dim=1)
    batch = torch.cat(batch_indices, dim=0)
    
    return {
        "reactiants_input_ids": reactiants_input_ids,
        "reactiants_attention_mask": reactiants_attention_mask,
        "products_input_ids": products_input_ids,
        "products_attention_mask": products_attention_mask,
        "labels": labels,
        "x": x,
        "edge_index": edge_index,
        "batch": batch
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = sklearn.metrics.accuracy_score(labels, preds)
    f1 = sklearn.metrics.f1_score(labels, preds, average="weighted")
    precision = sklearn.metrics.precision_score(labels, preds, average="weighted")
    recall = sklearn.metrics.recall_score(labels, preds, average="weighted")
    mcc = sklearn.metrics.matthews_corrcoef(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc
    }


def train_fold(fold, train_encodings, train_graphs, test_encodings, test_graphs, label_to_id, params):

    Epoch = params['Epoch']
    batch_size = params['batch_size']
    local_model_path = params['local_model_path']
    fusion_type = params['fusion_type']
    head_type = params['head_type']
    gnn_type = params['gnn_type']
    

    train_dataset = MultimodalSmilesDataset(train_encodings, train_graphs)
    test_dataset = MultimodalSmilesDataset(test_encodings, test_graphs)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn
    )
    
   
    logger.info(f"Fold {fold}: Loading multimodal model with {fusion_type} fusion, {head_type} head and {gnn_type} GNN...")
    model, tokenizer = load_multimodal_model(
        model_path=local_model_path,
        num_labels=len(label_to_id),
        fusion_type=fusion_type,
        head_type=head_type,
        gnn_type=gnn_type
    )
    logger.info(f"Fold {fold}: Model loaded successfully")
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Fold {fold}: Using device: {device}")
    
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    

    logger.info(f"Fold {fold}: Starting training for {Epoch} epochs...")
    for epoch in range(Epoch):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
           
            batch = {k: v.to(device) for k, v in batch.items()}
            
            
            outputs = model(**batch)
            loss = outputs["loss"]
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Fold {fold}: Epoch {epoch+1}/{Epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
        
      
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs["loss"]
                val_loss += loss.item()
          
                preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
       
        val_avg_loss = val_loss / len(test_loader)
        val_acc = sklearn.metrics.accuracy_score(all_labels, all_preds)
        val_f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted")
        
        logger.info(f"Fold {fold}: Epoch {epoch+1}/{Epoch}, Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    
    logger.info(f"Fold {fold}: Evaluating on test set...")
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            test_loss += loss.item()
            
           
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    
    test_avg_loss = test_loss / len(test_loader)
    test_acc = sklearn.metrics.accuracy_score(all_labels, all_preds)
    test_f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted")
    test_precision = sklearn.metrics.precision_score(all_labels, all_preds, average="weighted")
    test_recall = sklearn.metrics.recall_score(all_labels, all_preds, average="weighted")
    test_mcc = sklearn.metrics.matthews_corrcoef(all_labels, all_preds)
    
    logger.info(f"Fold {fold}: Test Results:")
    logger.info(f"Fold {fold}: Loss: {test_avg_loss:.4f}")
    logger.info(f"Fold {fold}: Accuracy: {test_acc:.4f}")
    logger.info(f"Fold {fold}: F1 Score: {test_f1:.4f}")
    logger.info(f"Fold {fold}: Precision: {test_precision:.4f}")
    logger.info(f"Fold {fold}: Recall: {test_recall:.4f}")
    logger.info(f"Fold {fold}: MCC: {test_mcc:.4f}")
    

    save_dir = f"/save/save_multimodal_fold{fold}_{Epoch}_{fusion_type}_{head_type}_{gnn_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Fold {fold}: Saving model to {save_dir}...")
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
    tokenizer.save_pretrained(save_dir)
    
    return {
        'loss': test_avg_loss,
        'accuracy': test_acc,
        'f1': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'mcc': test_mcc
    }


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "preprocessed2")
    label_map_path = os.path.join(cache_dir, "label_map.pkl")
    
  
    params = {
        'Epoch': 40,
        'batch_size': 64,
        'local_model_path': "zinc250k_v2_40k",
        'fusion_type': "attention",  #  "concat", "sum", "attention"
        'head_type': "mlp",  # "simple", "mlp"
        'gnn_type': "gin"   #"gcn", "gat", "gin"
    }
    
   
    logger.info(f"Loading label map from {label_map_path}...")
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
        label_to_id = label_map["label_to_id"]
        id_to_label = label_map["id_to_label"]
    
    
    num_folds = 5
    fold_results = []
    
   
    for fold in range(1, num_folds + 1):

    # for fold in [5]:
        logger.info(f"\n===== Processing Fold {fold}/{num_folds} =====")
        
     
        fold_dir = os.path.join(cache_dir, f"fold_{fold}")
        train_encodings_path = os.path.join(fold_dir, "train_encodings.pt")
        test_encodings_path = os.path.join(fold_dir, "test_encodings.pt")
        train_graphs_path = os.path.join(fold_dir, "train_graphs.pt")
        test_graphs_path = os.path.join(fold_dir, "test_graphs.pt")
        
        logger.info(f"Fold {fold}: Loading training encodings from {train_encodings_path}...")
        train_encodings = torch.load(train_encodings_path)
        test_encodings = torch.load(test_encodings_path)
        
        logger.info(f"Fold {fold}: Loading training graphs from {train_graphs_path}...")
        train_graphs = torch.load(train_graphs_path)
        test_graphs = torch.load(test_graphs_path)
        
        # 训练当前折
        fold_result = train_fold(
            fold=fold,
            train_encodings=train_encodings,
            train_graphs=train_graphs,
            test_encodings=test_encodings,
            test_graphs=test_graphs,
            label_to_id=label_to_id,
            params=params
        )
        
        fold_results.append(fold_result)
    
  
    logger.info("\n===== Cross-Validation Results =====")
    avg_results = {
        'loss': sum(r['loss'] for r in fold_results) / num_folds,
        'accuracy': sum(r['accuracy'] for r in fold_results) / num_folds,
        'f1': sum(r['f1'] for r in fold_results) / num_folds,
        'precision': sum(r['precision'] for r in fold_results) / num_folds,
        'recall': sum(r['recall'] for r in fold_results) / num_folds,
        'mcc': sum(r['mcc'] for r in fold_results) / num_folds
    }
    
    logger.info(f"Average Loss: {avg_results['loss']:.4f}")
    logger.info(f"Average Accuracy: {avg_results['accuracy']:.4f}")
    logger.info(f"Average F1 Score: {avg_results['f1']:.4f}")
    logger.info(f"Average Precision: {avg_results['precision']:.4f}")
    logger.info(f"Average Recall: {avg_results['recall']:.4f}")
    logger.info(f"Average MCC: {avg_results['mcc']:.4f}")
    
    logger.info("\n5-fold cross-validation completed successfully")
    

if __name__ == "__main__":
    main()