"""
A simplified E3NN protein model that uses a more basic approach to equivariance.
This version is more likely to work with older versions of e3nn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, LayerNorm
from torch_scatter import scatter


class SimpleE3nnLayer(nn.Module):
    def __init__(self, in_features, out_features, cutoff=10.0):
        """
        A simplified E3NN-inspired layer that respects basic geometric principles
        without requiring complex tensor product operations.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            cutoff: Maximum distance for considering interactions
        """
        super(SimpleE3nnLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.cutoff = cutoff
        
        # Networks for transforming node features
        self.node_projection = nn.Linear(in_features, out_features)
        
        # Networks for processing edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features + 4, out_features),  # node features + distance features
            nn.SiLU(),
            nn.Linear(out_features, out_features),
            nn.SiLU()
        )
        
        # Network for transforming distances (radial function)
        self.radial_func = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 4),  # 4 distance features
            nn.SiLU()
        )
        
    def forward(self, x, edge_index, pos):
        """
        Forward pass that respects basic geometric principles.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            pos: Node positions [num_nodes, 3]
            
        Returns:
            torch.Tensor: Updated node features [num_nodes, out_features]
        """
        source, target = edge_index
        
        # Compute relative positions
        rel_pos = pos[target] - pos[source]  # [num_edges, 3]
        
        # Compute distances
        edge_lengths = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
        
        # Apply distance cutoff
        edge_mask = edge_lengths < self.cutoff
        edge_mask = edge_mask.squeeze(-1)  # [num_edges]
        
        if not edge_mask.any():
            # Return transformed input if no edges within cutoff
            return self.node_projection(x)
        
        # Filter edges
        source = source[edge_mask]
        target = target[edge_mask]
        rel_pos = rel_pos[edge_mask]
        edge_lengths = edge_lengths[edge_mask]
        
        # Process distances through radial network
        radial_embedding = self.radial_func(edge_lengths)  # [num_edges, 4]
        
        # Get source node features
        source_features = x[source]  # [num_edges, in_features]
        
        # Combine node features with radial embedding
        edge_features = torch.cat([source_features, radial_embedding], dim=-1)
        
        # Process through edge network
        edge_features = self.edge_mlp(edge_features)  # [num_edges, out_features]
        
        # Scale by smooth cutoff function (optional)
        cutoff_factor = 1.0 - (edge_lengths / self.cutoff)**2
        cutoff_factor = torch.clamp(cutoff_factor, min=0.0, max=1.0)
        edge_features = edge_features * cutoff_factor
        
        # Aggregate to target nodes
        aggr_features = scatter(edge_features, target, dim=0, dim_size=x.size(0), reduce='sum')
        
        # Transform input features (residual path)
        transformed_features = self.node_projection(x)
        
        # Combine aggregated messages with transformed features
        return transformed_features + aggr_features


class SimpleE3nnProteinModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop=0.5, heads=4, k=None, add_self_loops=True):
        """
        A simplified protein structure model that respects basic geometric principles
        without requiring complex e3nn operations.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output predictions
            drop: Dropout probability
            heads: Number of attention heads (kept for API compatibility but not used)
            k: Number of neighbors (kept for API compatibility but not used)
            add_self_loops: Whether to add self-loops (kept for API compatibility)
        """
        super(SimpleE3nnProteinModel, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        
        # Initial projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # Equivariant-inspired layers
        self.conv1 = SimpleE3nnLayer(hidden_dim, hidden_dim, cutoff=10.0)
        self.conv2 = SimpleE3nnLayer(hidden_dim, hidden_dim, cutoff=10.0)
        self.conv3 = SimpleE3nnLayer(hidden_dim, hidden_dim, cutoff=10.0)
        
        # Layer norms
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)
        
        # Output MLPs
        self.lin0 = nn.Linear(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, edge_index, edge_attr, batch, pos):
        """
        Forward pass for the protein model.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_attr_dim] (ignored, kept for API compatibility)
            batch: Batch assignment [num_nodes]
            pos: Node positions [num_nodes, 3]
        
        Returns:
            tuple: (predictions, last_layer_features)
        """
        # Initial feature projection
        x = self.input_projection(x)
        
        # First layer
        x_conv1 = self.conv1(x, edge_index, pos)
        x = x + x_conv1  # Residual connection
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        
        # Second layer
        x_conv2 = self.conv2(x, edge_index, pos)
        x = x + x_conv2  # Residual connection
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        
        # Third layer
        x_conv3 = self.conv3(x, edge_index, pos)
        x = x + x_conv3  # Residual connection
        x = self.norm3(x, batch)
        
        # Global pooling - invariant to node ordering
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Final MLP layers
        x = F.dropout(x, p=self.drop, training=self.training)
        
        x = self.lin0(x)
        x = F.relu(x)
        
        x = self.lin1(x)
        x = F.relu(x)
        
        z = x  # Extract last layer features for compatibility
        
        x = self.lin(x)
        
        return x, z


# For backward compatibility with the original GAT model
GATModel = SimpleE3nnProteinModel
E3nnProteinModel = SimpleE3nnProteinModel