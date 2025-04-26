import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_scatter import scatter


class E3EquivariantGATLayer(MessagePassing):
    """
    E(3)-equivariant Graph Attention Layer that combines properties of both
    E(3) equivariance and attention mechanisms
    """
    def __init__(self, node_features, edge_features, hidden_dim, heads=4, dropout=0.1):
        super(E3EquivariantGATLayer, self).__init__(aggr="add")
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        
        # Attention mechanism
        self.query = nn.Linear(node_features, hidden_dim * heads)
        self.key = nn.Linear(node_features, hidden_dim * heads)
        self.value = nn.Linear(node_features, hidden_dim * heads)
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_features + 1, hidden_dim),  # +1 for distance
            nn.SiLU()
        ) if edge_features > 0 else None
        
        # Scalar networks (invariant to rotations)
        self.scalar_net = nn.Sequential(
            nn.Linear(node_features * 2 + hidden_dim, hidden_dim * heads),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Output transformations
        self.combine = nn.Linear(hidden_dim * heads, node_features)
        self.skip_connection = nn.Linear(node_features, node_features) if node_features != hidden_dim * heads else None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(node_features)
        
    def forward(self, x, pos, edge_index, edge_attr):
        # x: node features [N, node_features]
        # pos: node positions [N, 3]
        # edge_index: connectivity [2, E]
        # edge_attr: edge features [E, edge_features]
        
        # Skip connection
        identity = x
        
        # Compute message passing
        x = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        
        # Apply residual connection and normalization
        if self.skip_connection is not None:
            x = x + self.skip_connection(identity)
        else:
            x = x + identity
            
        return self.layer_norm(x)
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr, index):
        # Compute relative positions (equivariant feature)
        rel_pos = pos_j - pos_i  # [E, 3]
        dist = torch.norm(rel_pos, dim=1, keepdim=True)  # [E, 1]
        
        # Process edge attributes if available
        if self.edge_embedding is not None and edge_attr is not None:
            edge_features = self.edge_embedding(torch.cat([edge_attr, dist], dim=-1))
        else:
            edge_features = torch.zeros(dist.shape[0], self.hidden_dim, device=dist.device)
        
        # Compute queries, keys and values
        q = self.query(x_i).view(-1, self.heads, self.hidden_dim)  # [E, heads, hidden_dim]
        k = self.key(x_j).view(-1, self.heads, self.hidden_dim)    # [E, heads, hidden_dim]
        v = self.value(x_j).view(-1, self.heads, self.hidden_dim)  # [E, heads, hidden_dim]
        
        # Compute attention scores
        attention = (q * k).sum(dim=-1) / (self.hidden_dim ** 0.5)  # [E, heads]
        attention = F.softmax(attention, dim=0)  # Normalize over neighborhood
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Weight values by attention scores
        v = v * attention.unsqueeze(-1)  # [E, heads, hidden_dim]
        
        # Compute scalar message (invariant component)
        scalar_input = torch.cat([x_i, x_j, edge_features], dim=-1)
        scalar_msg = self.scalar_net(scalar_input).view(-1, self.heads, self.hidden_dim)  # [E, heads, hidden_dim]
        
        # Combine attention-weighted values with scalar message
        message = (v + scalar_msg).view(-1, self.heads * self.hidden_dim)  # [E, heads*hidden_dim]
        
        return message
    
    def update(self, aggr_out):
        # Process aggregated messages
        x = self.combine(aggr_out)
        return x


class CombinedE3GATModel(nn.Module):
    """Combined E(3)-equivariant GAT model for protein structure analysis"""
    
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, output_dim, 
                 num_layers=3, heads=4, dropout=0.1, pool='combined'):
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.pool = pool
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # E3 Equivariant GAT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(E3EquivariantGATLayer(
                node_features=hidden_dim,
                edge_features=edge_feature_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout
            ))
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, pos, edge_index, edge_attr, batch):
        # Initial feature projection
        x = self.input_proj(x)
        
        # Process through E3-equivariant GAT layers
        for layer in self.layers:
            x = layer(x, pos, edge_index, edge_attr)
        
        # Global pooling
        if self.pool == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.pool == 'max':
            pooled = global_max_pool(x, batch)
        else:  # combined
            mean_pooled = global_mean_pool(x, batch)
            max_pooled = global_max_pool(x, batch)
            pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            
        # Node embeddings for potential auxiliary tasks
        z = x
        
        # Final prediction
        out = self.output(pooled)
        
        return out, z


# Usage example:
# model = CombinedE3GATModel(
#     node_feature_dim=21,  # Amino acid one-hot + extra features
#     edge_feature_dim=5,   # Our enhanced edge features
#     hidden_dim=64,
#     output_dim=2,         # Binary classification example
#     num_layers=3,
#     heads=4,
#     dropout=0.1
# )