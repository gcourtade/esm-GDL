import pandas as pd
from graph import nodes, edges
from tqdm import tqdm
import numpy as np
from workflow.parameters_setter import ParameterSetter
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


def construct_graphs(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    construct_graphs
    :param workflow_settings:
    :param data: List (id, sequence itself, activity, label)
    :return:
        graphs_representations: list of Data
        labels: list of labels
        partition: identification of the old_data partition each instance belongs to
    """
    # nodes
    nodes_features, esm2_contact_maps = nodes.esm2_derived_features(workflow_settings, data)

    # edges
    # Old:
    # adjacency_matrices, weights_matrices, data = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    # New:  
    adjacency_matrices, weights_matrices, atom_coordinates_matrices, data = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(
                adjacency_matrix=adjacency_matrices[i],
                nodes_features=np.array(nodes_features[i], dtype=np.float32),
                weights_matrix=weights_matrices[i],
                label=data.iloc[i]['activity'] if 'activity' in data.columns else None,
                pos=atom_coordinates_matrices[i]  # <-- New: real 3D coordinates!
            ))

            progress.update(1)

    return graphs, data


def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, pos, eps=1e-6):

    """
    :param label: label
    :param adjacency_matrix: Adjacency matrix with shape (n_nodes, n_nodes)
    :param weights_matrix: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param nodes_features: node embedding with shape (n_nodes, n_node_features)
    :param eps: default eps=1e-6
    :return:
    """
    # Convert dense adjacency to sparse edge_index ([2, E]) plus optional per-edge weights ([E, F])
    A = torch.from_numpy(adjacency_matrix).to(torch.long)
    edge_index, _ = dense_to_sparse(A)             # edge_index: [2, E]

    # Build edge_attr if you supplied weights_matrix
    if weights_matrix is not None and weights_matrix.size:
        # weights_matrix: (N, N, F)
        W = torch.from_numpy(weights_matrix).float()
        row, col = edge_index
        edge_attr = W[row, col]                    # shape [E, F]
    else:
        edge_attr = None

    x = torch.from_numpy(nodes_features).float()    # [N, in_feats]
    y = torch.tensor([label], dtype=torch.int64) if label is not None else None
    pos = torch.from_numpy(pos).float()             # [N, 3]

    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                pos=pos)
    data.validate(raise_on_error=True)
    return data