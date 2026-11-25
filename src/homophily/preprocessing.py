from torch_geometric.utils import (to_undirected, 
                                   coalesce, 
                                   remove_self_loops,
                                   contains_self_loops, 
                                   is_undirected)

import torch

def preprocess_for_homophily(graph):
    """
    Preprocess graphs (undirected, no self loops, no duplicate edges) 
    """

    assert not contains_self_loops(graph.edge_index), f"{graph.name} contains self-loops!"
    assert is_undirected(graph.edge_index), f"{graph.name} is not undirected!" 
    
    edge_index_no_duplicates = coalesce(graph.edge_index)
    assert torch.equal(graph.edge_index, edge_index_no_duplicates), f"{graph.name} has duplicate edges!"

