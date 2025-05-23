import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch.nn import Linear
import torch


"""
Applying pyG lib
"""


class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, heads, k, add_self_loops):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = heads
        self.k = k # this isnt used here, but kept for compatibility
        self.add_self_loops = add_self_loops

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=heads, add_self_loops=add_self_loops)
        self.conv2 = GATConv(heads * hidden_dim, hidden_dim, heads=heads, add_self_loops=add_self_loops)
        self.conv3 = GATConv(heads * hidden_dim, hidden_dim, heads=heads, concat=False, add_self_loops=add_self_loops)

        self.norm1 = LayerNorm(heads * hidden_dim)
        self.norm2 = LayerNorm(heads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x, batch)

        # 2. Readout layer - Global Mean Pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        z = x  # extract last layer features

        x = self.lin(x)

        return x, z
