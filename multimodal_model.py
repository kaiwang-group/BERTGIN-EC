import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch_geometric.nn as gnn
from torch_geometric.nn import GINConv
from torch.nn import Linear

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob

    def forward(self, hidden_states):
        raise NotImplementedError("Subclasses must implement forward method.")

class SimpleClassificationHead(ClassificationHead):

    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super().__init__(hidden_size, num_labels, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

class MLPClassificationHead(ClassificationHead):

    def __init__(self, hidden_size, num_labels, dropout_prob=0.1, mlp_hidden_size=384):
        super().__init__(hidden_size, num_labels, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, mlp_hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(mlp_hidden_size, num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

class GraphFeatureExtractor(nn.Module):

    def __init__(self, in_channels=8, hidden_channels=128, out_channels=256, num_layers=3, gnn_type="gcn"):
        super().__init__()
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()
        
        if gnn_type == "gcn":
            # GCN layers
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        elif gnn_type == "gat":
          
            heads = 4

            assert hidden_channels % heads == 0, f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            assert out_channels % heads == 0, f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            
           
            self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
            
            self.convs.append(GATConv(hidden_channels, out_channels // heads, heads=heads))
        elif gnn_type == "gin":
            
            self.convs.append(GINConv(Linear(in_channels, hidden_channels), train_eps=True))
            
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(Linear(hidden_channels, hidden_channels), train_eps=True))
            
            self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
            
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.layernorm(x)
        return x

class MultimodalFusionModel(nn.Module):
   
    def __init__(self, model_path, num_labels, seq_hidden_size=768, graph_hidden_size=256, fusion_type="concat", head_type="mlp", gnn_type="gcn"):
        super().__init__()
        self.num_labels = num_labels
        self.fusion_type = fusion_type
        self.gnn_type = gnn_type
        
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        ).roberta 
        self.seq_hidden_size = seq_hidden_size
        
    
        self.graph_extractor = GraphFeatureExtractor(out_channels=graph_hidden_size, gnn_type=gnn_type)
        self.graph_hidden_size = graph_hidden_size
        
        
        if fusion_type == "concat":
            
            self.fusion_hidden_size = seq_hidden_size * 2 + graph_hidden_size
        elif fusion_type == "sum":
           
            self.fusion_hidden_size = seq_hidden_size * 2
            
            self.graph_proj = nn.Linear(graph_hidden_size, seq_hidden_size * 2)
        elif fusion_type == "attention":
            
            self.fusion_hidden_size = seq_hidden_size * 2
            
            self.graph_proj = nn.Linear(graph_hidden_size, seq_hidden_size * 2)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=seq_hidden_size * 2,
                num_heads=4,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
       
        if head_type == "simple":
            self.classifier = SimpleClassificationHead(self.fusion_hidden_size, num_labels)
        elif head_type == "mlp":
            self.classifier = MLPClassificationHead(self.fusion_hidden_size, num_labels)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
    
    def forward(self, reactiants_input_ids=None, reactiants_attention_mask=None, products_input_ids=None, products_attention_mask=None, x=None, edge_index=None, batch=None, labels=None):
        
        reactiants_outputs = self.transformer(reactiants_input_ids, attention_mask=reactiants_attention_mask, return_dict=True)
        reactiants_features = reactiants_outputs.last_hidden_state[:, 0, :]  
        
       
        products_outputs = self.transformer(products_input_ids, attention_mask=products_attention_mask, return_dict=True)
        products_features = products_outputs.last_hidden_state[:, 0, :]  
        

        seq_features = torch.cat([reactiants_features, products_features], dim=1)
        

        graph_features = self.graph_extractor(x, edge_index, batch)
        
        # Fusion
        if self.fusion_type == "concat":
            fusion_features = torch.cat([seq_features, graph_features], dim=1)
        elif self.fusion_type == "sum":
            graph_features_proj = self.graph_proj(graph_features)
            fusion_features = seq_features + graph_features_proj
        elif self.fusion_type == "attention":
      
            graph_features_proj = self.graph_proj(graph_features)
          
            combined = torch.stack([seq_features, graph_features_proj], dim=1)  
            attn_output, _ = self.attention(combined, combined, combined)
            fusion_features = attn_output.mean(dim=1)  
        
       
        logits = self.classifier(fusion_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

def load_multimodal_model(model_path, num_labels, fusion_type="concat", head_type="mlp", gnn_type="gcn"):
    model = MultimodalFusionModel(
        model_path=model_path,
        num_labels=num_labels,
        fusion_type=fusion_type,
        head_type=head_type,
        gnn_type=gnn_type
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
