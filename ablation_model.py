import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch_geometric.nn as gnn

class ClassificationHead(nn.Module):
    """Base class for classification heads."""
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
    
    def __init__(self, in_channels=8, hidden_channels=256, out_channels=512, num_layers=4, gnn_type="gat"):
        super().__init__()
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()
        
        if gnn_type == "gcn":
            
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
            
            from torch_geometric.nn import GINConv
            from torch.nn import Linear
            

            self.convs.append(GINConv(Linear(in_channels, hidden_channels), train_eps=True))

            for _ in range(num_layers - 2):
                self.convs.append(GINConv(Linear(hidden_channels, hidden_channels), train_eps=True))

            self.convs.append(GINConv(Linear(hidden_channels, out_channels), train_eps=True))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
            
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(out_channels)

        self.fc = nn.Linear(out_channels, out_channels)
        self.fc_activation = nn.ReLU()
        self.fc_norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.layernorm(x)

        x = self.fc(x)
        x = self.fc_activation(x)
        x = self.dropout(x)
        x = self.fc_norm(x)
        return x

class AblationMultimodalFusionModel(nn.Module):
    def __init__(self, model_path, num_labels, seq_hidden_size=768, graph_hidden_size=512, 
                 fusion_type="concat", head_type="mlp", gnn_type="gat",
                 use_reactiants=True, use_products_seq=True, use_products_graph=True):
        super().__init__()
        self.num_labels = num_labels
        self.fusion_type = fusion_type
        self.gnn_type = gnn_type
        self.use_reactiants = use_reactiants
        self.use_products_seq = use_products_seq
        self.use_products_graph = use_products_graph
        

        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        ).roberta  
        self.seq_hidden_size = seq_hidden_size
        

        if self.use_products_graph:
            self.graph_extractor = GraphFeatureExtractor(out_channels=graph_hidden_size, gnn_type=gnn_type)
            self.graph_hidden_size = graph_hidden_size
        
       
        self.calculate_fusion_size()
        
       
        if fusion_type == "concat":
            
            pass  
        elif fusion_type == "sum":

            if self.use_products_graph:
                self.graph_proj = nn.Linear(graph_hidden_size, self.fusion_hidden_size)
        elif fusion_type == "attention":
           
            if self.use_products_graph:
                self.graph_proj = nn.Linear(graph_hidden_size, self.fusion_hidden_size)

            if self.fusion_hidden_size > 0:
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.fusion_hidden_size,
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
    
    def calculate_fusion_size(self):

        seq_features_size = 0
        if self.use_reactiants:
            seq_features_size += self.seq_hidden_size
        if self.use_products_seq:
            seq_features_size += self.seq_hidden_size
        
        if self.fusion_type == "concat":
            if self.use_products_graph:
                self.fusion_hidden_size = seq_features_size + self.graph_hidden_size
            else:
                self.fusion_hidden_size = seq_features_size
        else:  
            if seq_features_size > 0:
                self.fusion_hidden_size = seq_features_size
            else:
               
                self.fusion_hidden_size = self.graph_hidden_size
    
    def forward(self, reactiants_input_ids=None, reactiants_attention_mask=None, 
                products_input_ids=None, products_attention_mask=None, 
                x=None, edge_index=None, batch=None, labels=None):
      
        features = []
        
 
        if self.use_reactiants:
            reactiants_outputs = self.transformer(reactiants_input_ids, 
                                                attention_mask=reactiants_attention_mask, 
                                                return_dict=True)
            reactiants_features = reactiants_outputs.last_hidden_state[:, 0, :]  
        
  
        if self.use_products_seq:
            products_outputs = self.transformer(products_input_ids, 
                                              attention_mask=products_attention_mask, 
                                              return_dict=True)
            products_features = products_outputs.last_hidden_state[:, 0, :]  
            features.append(products_features)
        

        if features:
            seq_features = torch.cat(features, dim=1)
        else:
         
            if reactiants_input_ids is not None:
                batch_size = reactiants_input_ids.shape[0]
                device = reactiants_input_ids.device
            elif products_input_ids is not None:
                batch_size = products_input_ids.shape[0]
                device = products_input_ids.device
            else:
               
                batch_size = batch.max().item() + 1
                device = x.device
            seq_features = torch.zeros(batch_size, self.fusion_hidden_size, device=device)
        
       
        if self.use_products_graph:
            graph_features = self.graph_extractor(x, edge_index, batch)
        
        # Fusion
        if self.fusion_type == "concat":
            if self.use_products_graph:
             
                if self.use_reactiants or self.use_products_seq:
                    fusion_features = torch.cat([seq_features, graph_features], dim=1)
                else:
                    
                    fusion_features = graph_features
            else:
                fusion_features = seq_features
        elif self.fusion_type == "sum":
            if self.use_products_graph:
                if self.use_reactiants or self.use_products_seq:
                    graph_features_proj = self.graph_proj(graph_features)
                    fusion_features = seq_features + graph_features_proj
                else:
                    
                    fusion_features = graph_features
            else:
                fusion_features = seq_features
        elif self.fusion_type == "attention":
            if self.use_products_graph:
                if self.use_reactiants or self.use_products_seq:
                    
                    graph_features_proj = self.graph_proj(graph_features)
                    
                    combined = torch.stack([seq_features, graph_features_proj], dim=1)  
                    attn_output, _ = self.attention(combined, combined, combined)
                    fusion_features = attn_output.mean(dim=1)  
                else:
                    
                    fusion_features = graph_features
            else:
                fusion_features = seq_features
        
     
        logits = self.classifier(fusion_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

def load_ablation_model(model_path, num_labels, fusion_type="concat", head_type="mlp", gnn_type="gcn",
                       use_reactiants=True, use_products_seq=True, use_products_graph=True):
    """Load ablation model with specified fusion type, classification head and GNN type."""
    model = AblationMultimodalFusionModel(
        model_path=model_path,
        num_labels=num_labels,
        fusion_type=fusion_type,
        head_type=head_type,
        gnn_type=gnn_type,
        use_reactiants=use_reactiants,
        use_products_seq=use_products_seq,
        use_products_graph=use_products_graph
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
