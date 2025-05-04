import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, LayerNorm
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import Sequential, FullyConnectedNet
from e3nn.o3 import Irreps


class E3nnConvLayer(nn.Module):
    def __init__(self, in_features, out_features, lmax=1, radius=10.0):
        super(E3nnConvLayer, self).__init__()
        
        # Define irreps for scalars (l=0) and vectors (l=1)
        self.irreps_in = Irreps(f"{in_features}x0e")  # Scalar features
        self.irreps_out = Irreps(f"{out_features}x0e")  # Output scalar features
        
        # Define irreps for the edge attributes (distance embedding)
        edge_attr_irreps = Irreps("16x0e")
        
        # Define the spherical harmonics for angular information
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        
        # Network to create edge features from scalar distance
        self.edge_embedding = FullyConnectedNet(
            [1, 16, 16, 16], 
            acts=[torch.nn.SiLU(), torch.nn.SiLU(), torch.nn.SiLU()]
        )
        
        # Tensor product for combining node features with spherical harmonics
        self.tp = o3.FullTensorProduct(
            self.irreps_in, 
            self.sh_irreps,
            irreps_out=self.irreps_out
        )
        
        # MLP for applying after tensor product
        self.post_tp_mlp = FullyConnectedNet(
            [out_features, out_features, out_features],
            acts=[torch.nn.SiLU(), torch.nn.SiLU()]
        )
        
        self.radius = radius
    
    def forward(self, x, edge_index, pos):
        """
        x: Node features [num_nodes, in_features]
        edge_index: Connectivity [2, num_edges]
        pos: Node positions [num_nodes, 3]
        """
        source, target = edge_index
        
        # Compute relative positions (vectors pointing from source to target)
        rel_pos = pos[target] - pos[source]  # [num_edges, 3]
        
        # Compute distances
        edge_lengths = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
        
        # Create edge mask for edges within radius
        edge_mask = edge_lengths < self.radius
        edge_mask = edge_mask.squeeze(-1)  # [num_edges]
        
        if not edge_mask.any():
            # If no edges within radius, return zeros
            return torch.zeros_like(x)
        
        # Filter edges
        source = source[edge_mask]
        target = target[edge_mask]
        rel_pos = rel_pos[edge_mask]
        edge_lengths = edge_lengths[edge_mask]
        
        # Normalize directions for spherical harmonics
        directions = rel_pos / (edge_lengths + 1e-8)  # [num_edges, 3]
        
        # Compute spherical harmonics of the directions
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, 
            directions, 
            normalize=True
        )  # [num_edges, sh_dim]
        
        # Compute radial (distance) embedding
        radial_embedding = self.edge_embedding(edge_lengths)  # [num_edges, 16]
        
        # Gather source node features
        source_features = x[source]  # [num_edges, in_features]
        
        # Apply tensor product to combine features with spherical harmonics
        tp_out = self.tp(source_features, edge_sh)  # [num_edges, out_features]
        
        # Apply radial embedding as a gating mechanism
        gated_features = tp_out * radial_embedding.sum(-1, keepdim=True)
        
        # Apply MLP
        edge_features = self.post_tp_mlp(gated_features)  # [num_edges, out_features]
        
        # Aggregate messages at target nodes
        out = scatter(edge_features, target, dim=0, dim_size=x.size(0), reduce='sum')
        
        return out


class E3nnProteinModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop=0.5, heads=4, k=None, add_self_loops=True):
        """
        A protein structure model that preserves E(3) equivariance.
        
        Parameters match the original GATModel for drop-in replacement.
        
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output predictions
            drop: Dropout probability
            heads: Number of attention heads (kept for API compatibility but not used)
            k: Number of neighbors (kept for API compatibility but not used)
            add_self_loops: Whether to add self-loops (kept for API compatibility)
        """
        super(E3nnProteinModel, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        
        # Initial projection of node features
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # E3nn equivariant layers
        self.conv1 = E3nnConvLayer(hidden_dim, hidden_dim, lmax=1, radius=10.0)
        self.conv2 = E3nnConvLayer(hidden_dim, hidden_dim, lmax=1, radius=10.0)
        self.conv3 = E3nnConvLayer(hidden_dim, hidden_dim, lmax=1, radius=10.0)
        
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
    
    def forward(self, x, edge_index, edge_attr, batch, pos=None):
        """
        Forward pass for the E3nn protein model.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_attr_dim] (ignored, kept for API compatibility)
            batch: Batch assignment [num_nodes]
            pos: Node positions [num_nodes, 3] (required for E3nn processing but wasn't in original API)
                 If not provided, will attempt to extract from edge_attr assuming it contains position info
        
        Returns:
            tuple: (predictions, last_layer_features)
        """
        # If positions aren't provided, we need to extract them from somewhere
        if pos is None:
            # This is a workaround assuming that edge_attr might contain positional information
            # In a real implementation, you would need to ensure that 3D coordinates are provided
            raise ValueError("Node positions (pos) must be provided for E3nn models")
        
        # Initial feature projection
        x = self.input_projection(x)
        
        # First E3nn layer
        x_conv1 = self.conv1(x, edge_index, pos)
        x = x + x_conv1  # Residual connection
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        
        # Second E3nn layer
        x_conv2 = self.conv2(x, edge_index, pos)
        x = x + x_conv2  # Residual connection
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        
        # Third E3nn layer
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
GATModel = E3nnProteinModel
