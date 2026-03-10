import os
import torch
import logging
import pickle
import sklearn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from multimodal_model import MultimodalFusionModel, load_multimodal_model
from train import MultimodalSmilesDataset, multimodal_collate_fn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trained_model():

    logger.info("Testing trained multimodal model...")
    
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
   
    Epoch = 5
    fusion_type = "attention"  
    head_type = "mlp"  
    gnn_type = "gat"  
    

    original_model_path = "zinc250k_v2_40k"

    trained_model_path = f"save_multimodal_{Epoch}_{fusion_type}_{head_type}_{gnn_type}"

    cache_dir = os.path.join(script_dir, "preprocessed")
    test_encodings_path = os.path.join(cache_dir, "test_encodings.pt")
    test_graphs_path = os.path.join(cache_dir, "test_graphs.pt")
    label_map_path = os.path.join(cache_dir, "label_map.pkl")
    

    logger.info(f"Loading label map from {label_map_path}...")
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
        label_to_id = label_map["label_to_id"]
        id_to_label = label_map["id_to_label"]
    num_labels = len(label_to_id)
    

    logger.info(f"Loading test encodings from {test_encodings_path}...")
    test_encodings = torch.load(test_encodings_path)
    test_graphs = torch.load(test_graphs_path)
    

    test_dataset = MultimodalSmilesDataset(test_encodings, test_graphs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=multimodal_collate_fn
    )
    

    logger.info(f"Loading trained model from {trained_model_path}...")
    

    model, tokenizer = load_multimodal_model(
        model_path=original_model_path,
        num_labels=num_labels,
        fusion_type=fusion_type,
        head_type=head_type,
        gnn_type=gnn_type
    )
    

    model_weights_path = os.path.join(trained_model_path, "model.pt")
    model.load_state_dict(torch.load(model_weights_path))
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    

    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    
    logger.info(f"Evaluating on test set...")
    
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
    test_f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_precision = sklearn.metrics.precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_recall = sklearn.metrics.recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    test_mcc = sklearn.metrics.matthews_corrcoef(all_labels, all_preds)
    

    logger.info("Test Results:")
    logger.info(f"Loss: {test_avg_loss:.4f}")
    logger.info(f"Accuracy: {test_acc:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall: {test_recall:.4f}")
    logger.info(f"MCC: {test_mcc:.4f}")
 
    try:
        conf_matrix = sklearn.metrics.confusion_matrix(all_labels, all_preds)
        logger.info(f"Confusion Matrix shape: {conf_matrix.shape}")
    except Exception as e:
        logger.warning(f"Failed to compute confusion matrix: {e}")
    
    logger.info("Test completed successfully!")
    
    return {
        "loss": test_avg_loss,
        "accuracy": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "mcc": test_mcc
    }


if __name__ == "__main__":
    results = test_trained_model()
    logger.info(f"Final test results: {results}")