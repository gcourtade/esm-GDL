import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from e3nn.o3 import Irreps, spherical_harmonics, FullyConnectedTensorProduct


class EquivariantConv(MessagePassing):
    """
    Equivariant edge-message-passing layer using e3nn TensorProducts.
    """
    def __init__(self, irreps_in: Irreps, irreps_sh: Irreps, irreps_out: Irreps, aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        # Tensor product: combine node features with edge spherical harmonics
        self.tp = FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_out)
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = irreps_out

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor, pos: torch.Tensor):
        # x: [N, C_in] flattened repr of irreps_in
        # pos: [N, 3] node coordinates
        # Compute relative displacement for each edge
        row, col = edge_index
        edge_vec = pos[row] - pos[col]                           # [E, 3]
        # Compute spherical harmonics Y^l_m on each edge
        edge_attr = spherical_harmonics(self.irreps_sh, edge_vec, normalize=True)
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j: [E, C_in], edge_attr: [E, C_sh]
        # Combine via TensorProduct -> [E, C_out]
        return self.tp(x_j, edge_attr)


class E3nnProteinModel(nn.Module):
    """
    Equivariant graph classification model. Replace your GAT_1.py with this file.
    Expects your Data objects to have `pos` attribute (node coords, shape [N,3]).
    Forward signature is identical, plus requires `pos` argument.
    """
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        drop: float = 0.0,
        heads: int = 1,
        k: int = None,
        add_self_loops: bool = True,
        lmax: int = 2,
    ):
        super().__init__()
        # irreps for input, spherical harmonics, hidden, and output
        self.irreps_in    = Irreps(f"{node_feature_dim}x0e")       # scalar ESM embeddings
        self.irreps_sh    = Irreps.spherical_harmonics(lmax)
        self.irreps_hidden = Irreps(f"{hidden_dim}x0e")          # keep invariants
        self.irreps_out   = self.irreps_hidden

        # Initial linear map from raw scalars to hidden
        self.lin_in = nn.Linear(node_feature_dim, hidden_dim)
        # Equivariant conv layers
        self.conv1 = EquivariantConv(self.irreps_hidden, self.irreps_sh, self.irreps_hidden)
        self.conv2 = EquivariantConv(self.irreps_hidden, self.irreps_sh, self.irreps_hidden)
        self.conv3 = EquivariantConv(self.irreps_hidden, self.irreps_sh, self.irreps_hidden)

        # Classification MLP
        self.dropout = drop
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_in.weight)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, edge_index, edge_attr=None, batch=None, pos=None):
        # Support both (data.x, data.edge_index, ..., data.pos) and model(data) if data is Data
        if isinstance(x, Data):
            data = x
            x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos
        if pos is None:
            raise ValueError("Equivariant model requires `pos` (node coordinates) in forward call.")

        # 1. Initial scalar embed
        h = self.lin_in(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # 2. Equivariant message passing
        # Flatten and keep invariant channels
        irreps_h = self.irreps_hidden
        # Expand h to match irreps representation dims
        h = h.view(-1, irreps_h.dim)
        h = self.conv1(h, edge_index, pos)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, pos)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index, pos)

        # 3. Global pooling over nodes
        # h is scalar invariants -> shape [N, hidden_dim]
        h = h.view(-1, self.irreps_hidden.dim)
        h_pool = global_mean_pool(h, batch)

        # 4. Classification
        out = self.classifier(h_pool)
        # z: penultimate features
        z = h_pool
        return out, z