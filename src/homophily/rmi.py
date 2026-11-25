from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from torch_geometric.utils import is_undirected, to_networkx
import networkx as nx
import community
import torch

def louvain_community(graph, resolution=1.0):

    G = to_networkx(graph, to_undirected=True)

    communities = community.best_partition(G, random_state=42, resolution=resolution)

    community_ids = torch.empty(graph.num_nodes, dtype=torch.long)

    for node, comm in communities.items():
        community_ids[node] = comm

    return community_ids 


def rmi(graph, visible_nodes=None): 
    """
    What fraction of the uncertainty in L is explained by C?”
    rmi = I(L;C)/H(L)

    Community detection performed on the whole graph!
    """

    communities = louvain_community(graph)
    labels = graph.y.cpu()
    
    if visible_nodes is not None:
        visible = visible_nodes.cpu()
        communities = communities[visible]
        labels = labels[visible]

    # I(L;C) = H(L)-H(L|C)
    mi = mutual_info_score(labels.numpy(), communities.numpy())

    label_counts = torch.bincount(labels)
    h_l = entropy(label_counts.float().numpy())

    if h_l == 0:
        return float("nan")

    return round(mi / h_l, 4)