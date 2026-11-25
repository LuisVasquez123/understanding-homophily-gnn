import torch
from torch_geometric.utils import is_undirected, to_networkx
from torch_geometric.utils import degree as get_degree
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
import torch.nn.functional as F


def adjacency_list(graph):
    """
    Optimization for get_neighbors()
        adj[0] = [1, 3, 4]
        adj[1] = [0, 2]
        adj[2] = [1]
    """
    adj = [[] for _ in range(graph.num_nodes)]
    src = graph.edge_index[0].tolist()
    dst = graph.edge_index[1].tolist()

    for s, d in zip(src, dst):
        adj[s].append(d)

    return adj



def verify_graph_is_connected(graph):

    print(f"PYG: {graph.edge_index.size(1)} edges.")
    G = to_networkx(graph, to_undirected=True) # (u,v)
    print(f"Networkx (class: {G.__class__.__name__}): {G.number_of_edges()} edges.")

    if not nx.is_connected(G):
        raise ValueError(f"{graph} is not connected!")



def get_neighbors(graph, v, same_label=False):
    src = graph.edge_index[0] 
    dst = graph.edge_index[1]

    mask = (src == v)
    nbs = dst[mask]

    return nbs



def neighbor_distribution_matrix(data):
    """
    D^N ∈ NxC 
    D^N[u,:] : u's neighbor distribution over classes (probs., sum=1)
    """
    
    adj = adjacency_list(data)
    num_classes = len(data.y.unique())
    num_nodes = data.num_nodes
    
    D = torch.zeros((num_nodes, num_classes), dtype=torch.float)

    for v in range(data.num_nodes):
        neighbors = adj[v]
        degree = len(neighbors)

        if degree == 0:
            continue

        for u in neighbors:
            y_u = data.y[u]
            D[v,y_u] += 1

        D[v,:] /= degree # probs.
    
    return D



# ================================================== LABEL HOMOPHILY MEASURES ==================================================
# Node (h1)
def node_homophily(graph):
    
    total = 0
    valid_nodes = 0 # non-isolated
    adj = adjacency_list(graph)

    for v in range(graph.num_nodes):
        neighbors = adj[v]
        neighbors_same_label = {u for u in neighbors if graph.y[v] == graph.y[u]}
        degree = len(neighbors)
        
        # don't count if isolated
        if degree == 0:
            continue
        
        v_homophily = len(neighbors_same_label) / degree
        total += v_homophily
        # only average over connected nodes
        valid_nodes += 1

    if valid_nodes == 0:
        return float("nan")
    
    homophily = total / valid_nodes

    return round(homophily, 4)


# Node (h2)
def node_homophily_h2(graph, visible_nodes=None):
    '''
    2-hop neighbors: -unique- (no weighting if multiple walks) nodes -strictly- at distance 2, except self
    visible_nodes = graph.train_mask.nonzero(as_tuple=True)[0] defined beforehand if computing on subgraph
    '''
    if visible_nodes is None:
        visible_nodes = graph.train_mask.nonzero(as_tuple=True)[0]

    total = 0
    valid_nodes = 0 
    adj = adjacency_list(graph)
    visible_nodes = set(visible_nodes.tolist())

    for v in visible_nodes:
        neighbors = adj[v]
        neighbors_2 = set() 

        for n in neighbors:    
            neighbors_2.update(adj[n]) 
        
        neighbors_2.discard(v) 
        neighbors_2.difference_update(neighbors) 
        neighbors_2.intersection_update(visible_nodes) 
        neighbors_2_same_label = {u for u in neighbors_2 if graph.y[v] == graph.y[u]}
        
        degree_2 = len(neighbors_2)
        
        if degree_2 == 0:
            continue
        
        v_homophily = len(neighbors_2_same_label) / degree_2
        total += v_homophily
        # node is valid if it has at least one 2-hop neighbor (else x/0)
        valid_nodes += 1

    if valid_nodes == 0:
        return float("nan")
    
    homophily = total / valid_nodes

    return round(homophily, 4)


# Edge
def edge_homophily(graph):

    count = 0
    num_edges = graph.edge_index.shape[1]
    
    if num_edges == 0:
        return float("nan")

    for i in range(num_edges):
        if graph.y[graph.edge_index[0][i]] == graph.y[graph.edge_index[1][i]]:
            count += 1

    homophily = count / num_edges
    return round(homophily, 4)


# Class
def class_homophily(graph):
    
    num_classes = len(graph.y.unique())
    num_valid_classes = 0
    adj = adjacency_list(graph)

    node_degrees = get_degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    non_isolated_mask = node_degrees > 0
    n = non_isolated_mask.sum().item() 

    if n == 0:
        return float("nan")

    excess_homophilies = []
    for k in graph.y.unique():
        
        nodes_class_k = torch.where(graph.y == k)[0]
        n_k = 0

        sum_neighbors_class_k = 0
        degree_class_k = 0
        for v in nodes_class_k:
            
            v = v.item()
            neighbors = adj[v]
            degree = len(neighbors)
            
            # don't count if isolated
            if degree == 0:
                continue
            
            n_k += 1
            degree_class_k += len(neighbors)
            neighbors_same_label = [u for u in neighbors if graph.y[v] == graph.y[u]]
            sum_neighbors_class_k += len(neighbors_same_label)

        # class has only isolated edges
        if degree_class_k == 0:
            continue

        num_valid_classes += 1

        excess_homophily_class_k = (sum_neighbors_class_k / degree_class_k) - (n_k / n)
        excess_homophily_class_k = max(0, excess_homophily_class_k)
        
        excess_homophilies.append(excess_homophily_class_k)

    
    if num_valid_classes == 1:
        return float("nan")
    
    homophily = (1 / (num_valid_classes - 1)) * sum(excess_homophilies)
    return round(homophily, 4)


# Adjusted
def adjusted_homophily(graph):

    if graph.num_edges == 0:
        return float("nan")
    
    h_edge = edge_homophily(graph)
    
    adj = adjacency_list(graph)

    expected_homophily = 0
    for k in graph.y.unique():
        
        nodes_class_k = torch.where(graph.y == k)[0]

        degree_class_k = 0
        for v in nodes_class_k:
            v = v.item()
            neighbors = adj[v]
            degree_class_k += len(neighbors)
        
        assert is_undirected(graph.edge_index), "Graph is not undirected!"
        expected_homophily += (degree_class_k ** 2) / ((graph.num_edges) ** 2) # (undirected) graph.num_edges = 2|E|

    # Modularity
    Q = h_edge - expected_homophily

    if 1 - expected_homophily == 0:
        return float("nan")
    
    homophily = Q / (1 - expected_homophily)

    return round(homophily, 4)


# Unbiased 
def unbiased_homophily(graph):
    m = len(graph.y.unique())
    C_G = torch.zeros((m, m), dtype=torch.float)

    # Verify all classes sequential
    assert torch.all(graph.y.unique() == torch.arange(m)), "Classes not 0..m-1, mapping required!"

    if graph.num_edges == 0:
        return float("nan")
    
    for i in range(graph.num_edges):
        u = graph.edge_index[0][i]
        v = graph.edge_index[1][i]

        cu = graph.y[u]
        cv = graph.y[v]

        C_G[cu][cv] += 1

    C_G *= 1 / (graph.num_edges / 2)
    
    num = 0
    den = 0
    for i in range(m):
        
        for j in range(i+1, m):
    
            num += (torch.sqrt(C_G[i][i] * C_G[j][j])) - C_G[i][j]
            den += (torch.sqrt(C_G[i][i] * C_G[j][j])) + C_G[i][j]

    if den == 0:
        return float("nan")
    
    homophily = num / den

    return round(homophily.item(), 4)



# ================================================== FEATURE HOMOPILY MEASURES ==================================================
# GE
def generalized_edge_homophily(data):

    x = data.x.float()
    src, dst = data.edge_index_unidirectional
    x_src = x[src]
    x_dst = x[dst]

    # avg. similarity of features across all edges 
    similarities = F.cosine_similarity(x_src, x_dst, dim=1)

    # similarity across edges edges with same/diff labels?
    mask = (data.y[src] == data.y[dst]) 

    return {
        "sim_all": similarities.mean().item(),
        "sim_same_label": similarities[mask].mean().item(),
        "sim_diff_label": similarities[~mask].mean().item()
    }


# FA-p-dim
def feature_assortativity_per_dimension(data):

    x = data.x.float()
    src, dst = data.edge_index_unidirectional
    x_src = x[src]
    x_dst = x[dst]

    r_features = []
    
    for f in range(x.size(1)):
        a = x_src[:, f].cpu().numpy()
        b = x_dst[:, f].cpu().numpy()

        if np.isclose(np.std(a), 0) or np.isclose(np.std(b), 0):
            continue

        r, p = pearsonr(a, b)
        
        if not np.isnan(r):
            r_features.append(r)

    if len(r_features) == 0:
        return {
        "r_mean": float("nan"),
        "r_mean_abs": float("nan"),
        "valid_features": len(r_features)
        } 
    
    return {
        "r_mean": float(np.mean(r_features)),
        "r_mean_abs": float(np.mean(np.abs(r_features))),
        "valid_features": len(r_features)
    } 


# FA-flat
def feature_assortativity_flattened(data):
  
    x = data.x.float()
    src, dst = data.edge_index_unidirectional

    x_src = x[src]
    x_dst = x[dst]

    a = x_src.flatten().cpu().numpy()
    b = x_dst.flatten().cpu().numpy()

    if np.isclose(np.std(a), 0) or np.isclose(np.std(b), 0):
        r = float("nan")
    else: 
        r, p = pearsonr(a, b)

    return r


# LS
def local_similarity(data, sim):

    total = 0
    valid_nodes = 0 # non-isolated
    adj = adjacency_list(data)
    x = data.x.float()

    for v in range(data.num_nodes):
        neighbors = adj[v]
        degree = len(neighbors)
        x_v = x[v]

        # don't count if isolated
        if degree == 0:
            continue

        # nb similarities
        sum_similarities = 0 
        for u in neighbors:
            x_u = x[u]
            if sim == "cos":
                simimlarity = F.cosine_similarity(x_v, x_u, dim=0) 
            elif sim == "euc":
                simimlarity = -torch.norm(x_v - x_u)
            else: 
                raise NotImplementedError(f"Metric not recognized: {sim}")
            sum_similarities += simimlarity.item()
        
        v_homophily = (1/degree) * sum_similarities

        total += v_homophily

        # only average over connected nodes
        valid_nodes += 1

    if valid_nodes == 0:
        return float("nan")
    
    homophily = total / valid_nodes

    return round(homophily, 4)


# ================================================== STRUCTURE HOMOPILY MEASURES ==================================================
# D-asst
def degree_assortativity(data):

    src, dst = data.edge_index_unidirectional
    degrees = get_degree(data.edge_index[0], num_nodes=data.num_nodes)

    a = degrees[src].cpu().numpy()
    b = degrees[dst].cpu().numpy()

    r, p = pearsonr(a, b)
    
    return r


# LCC-asst
def lcc_assortativity(data):

    src, dst = data.edge_index_unidirectional

    # LCC
    G = to_networkx(data) 

    lccs_dict = nx.clustering(G)

    lccs = [lccs_dict[i] for i in range(data.num_nodes)]
    lccs = torch.tensor(lccs, dtype=float)

    a = lccs[src].cpu().numpy()
    b = lccs[dst].cpu().numpy()

    r, p = pearsonr(a, b)
    
    return r



# NS (CCNS)
def neighborhood_similarity(data):
    """
    Neighborhood Similarity NS ∈ [0..inf]: avg. sim. intra-class nodes nbh dist / avg. sim. inter-class nodes nbh dist
        S ∈ NxN
        S[u,v] = cos(D_u, D_v)
        NS = E(cos(D_u, D_v) | y_u=y_v) / E(cos(D_u, D_v) | y_u!=y_v)

    """
    
    adj = adjacency_list(data)
    D = neighbor_distribution_matrix(data)
    y = data.y

    # collect inter-class and intra-class similarities from S
    intra_class_similarities = [] 
    inter_class_similarities = [] 
    for v in range(data.num_nodes):
        
        if len(adj[v]) == 0: continue

        D_v = D[v,:]

        for u in range(data.num_nodes):
            
            if len(adj[u]) == 0: continue
            if u == v: continue 

            D_u = D[u,:]

            sim = F.cosine_similarity(D_v, D_u, dim=0) # S[v, u]

            if y[v] == y[u]:
                intra_class_similarities.append(sim.item())
            else:
                inter_class_similarities.append(sim.item())

    expected_intra_class_similarity = sum(intra_class_similarities) / len(intra_class_similarities)
    expected_inter_class_similarity = sum(inter_class_similarities) / len(inter_class_similarities)

    if expected_inter_class_similarity == 0:
        return float("nan")
    
    # NS
    homophily = expected_intra_class_similarity / expected_inter_class_similarity

    return round(homophily, 4)
           