import matplotlib.pyplot as plt
from torch_geometric.utils import (
    subgraph,
    contains_self_loops, 
    to_networkx,
    degree,
    is_undirected,
    )
from torch_geometric.data import Data
import networkx as nx
import torch

def class_compatibility_matrix(data, num_classes):
    """
    Hij = P(nb class = j | node class = i) -- so rows sum to 1
    """

    src, dst = data.edge_index
    y = data.y

    H_counts = torch.zeros((num_classes, num_classes))

    for i, j in zip(src, dst):
        H_counts[y[i], y[j]] += 1 # symmetric

    rows_sums = H_counts.sum(dim=1, keepdim=True)

    H = H_counts / rows_sums # not symmetric

    return H, H_counts



def to_unidirectional(edge_index):
    """
    Each edge stored only once: [u, v], [v, u] --> [u, v]
    """
    assert is_undirected(edge_index), "Graph must be undirected"

    src, dst = edge_index
    mask = (src < dst)
    new_edge_index = torch.stack([src[mask], dst[mask]], dim=0)

    return new_edge_index



def connectivity_stats(data):
    """
    PYG -> NetworkX for connectivity functions
    """
    G = to_networkx(data, to_undirected=True)

    connected_components = list(nx.connected_components(G))

    num_connected_components = len(connected_components)
    largest_connected_component_size = max(len(c) for c in connected_components)
    is_connected = (num_connected_components == 1)

    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

    num_isolated_nodes = int((node_degrees == 0).sum().item())

    return {
        "Is connected": is_connected,
        "Number of connected components": num_connected_components, 
        "Largest connected component size": largest_connected_component_size,
        "Has isolated nodes": data.has_isolated_nodes(),
        "Number of isolated nodes": num_isolated_nodes,
    }



def generate_pyg_subgraph(graph, visible_nodes, relabel_nodes=True):
    """
    From PYG graph to label-induced subgraph G_L=G[V_L]
        Nodes are renamed
        Isolated nodes are inferred from new edge_index
    """
    subgraph_edge_index, subgraph_edge_attr = subgraph(subset=visible_nodes, 
                                                       edge_index=graph.edge_index, 
                                                       edge_attr=getattr(graph, "edge_attr", None),
                                                       relabel_nodes=relabel_nodes)

    subgraph_x = graph.x[visible_nodes] if getattr(graph, "x", None) is not None else None
    subgraph_y = graph.y[visible_nodes] if getattr(graph, "y", None) is not None else None

    return Data(x=subgraph_x, 
                y=subgraph_y, 
                edge_index=subgraph_edge_index, 
                edge_attr=subgraph_edge_attr,
                num_nodes=len(visible_nodes))

