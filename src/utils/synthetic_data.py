from torch_geometric.data import Data
from torch_geometric.utils import coalesce, degree
import numpy as np
import torch
from . import class_compatibility_matrix
from src.homophily import edge_homophily
from src.visualization import plot_csbm_h, plot_csbm_h_features, plot_graph, plot_data_projection
import networkx as nx

class CSBM_H_2:
    """
    n0, n1: class sizes
    mu0, mu1: mean vectors of class features
    sigma0, sigma1: variance of class features
    d0, d1: class (out-)degrees
    h: (edge) homohpily

    Binary Compatibility Matrix:
    H = |  h  1-h |
        | 1-h  h  |

    Adapted from: "When Do Graph Neural Networks Help with Node Classification? 
        Investigating the Impact of Homophily Principle on Node Distinguishability"
    """

    def __init__(self, n0, n1, mu0, mu1, sigma0, sigma1, d0, d1):
        
        assert len(mu0) == len(mu1)
        self.num_classes = 2
        self.num_nodes = n0 + n1
        self.degrees = [d0, d1]
        self.dim = len(mu0)

        # csbm-h
        self.n0 = n0
        self.n1 = n1
        self.mu0 = torch.as_tensor(mu0, dtype=torch.float)
        self.mu1 = torch.as_tensor(mu1, dtype=torch.float)
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.d0 = d0
        self.d1 = d1

        self.X = None
        self.y = None

    def generate_labels(self):

        # y0 ∈ R^n0
        y0 = torch.zeros(self.n0, dtype=torch.long)
        y1 = torch.ones( self.n1, dtype=torch.long)
        self.y = torch.cat([y0, y1]) 
        return self.y
    
    def generate_features(self):

        # x0 ~ N(mu0, sigma0*I) ∈ R^n0xdim
        x0 = torch.distributions.MultivariateNormal(
            loc=self.mu0, 
            covariance_matrix=self.sigma0*torch.eye(self.dim),
            ).sample((self.n0,))
        x1 = torch.distributions.MultivariateNormal(
            loc=self.mu1, 
            covariance_matrix=self.sigma1*torch.eye(self.dim),
            ).sample((self.n1,))
        self.X = torch.vstack([x0,x1])  
        return self.X
    
    def generate_compatibility_matrix(self, h):

        assert 0 <= h <= 1, "h must be between 0 and 1!"

        return torch.tensor([
            [h, 1-h],       # C0
            [1-h, h]        # C1
        ], dtype=torch.float)
    
    def generate_edge_index(self, h, directed=False):
    
        assert self.y is not None, "Labels (y) have not been created!"
        
        H = self.generate_compatibility_matrix(h)

        class_nodes = {}
        for c in range(self.num_classes):
            class_nodes[c] = torch.where(self.y == c)[0]

        edges = []
        for u in range(self.num_nodes):
            
            # src node 
            y_u = int(self.y[u])
            degree_u = self.degrees[y_u]

            # sample degree_u nbs
            for _ in range(degree_u):
                
                class_probs = H[y_u]
                # sample nb from class probabilities
                y_v = torch.multinomial(class_probs, num_samples=1, replacement=True).item()
                candidates = class_nodes[y_v]
                
                if len(candidates) == 0 or (len(candidates) == 1 and candidates[0].item() == u): 
                    continue                
                
                while True:
                    idx = torch.randint(len(candidates), (1, )).item()
                    v = candidates[idx] 
                    if v != u: break # no self loops
            
                edges.append((u, v)) 
               
                if not directed: edges.append((v, u)) 

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = coalesce(edge_index)

        return edge_index

    def generate_data(self, h, directed=False):

        assert self.X is not None and self.y is not None, \
            f"Features (X) and/or labels (y) have not been created!"
        edge_index = self.generate_edge_index(h, directed)

        return Data(x=self.X, edge_index=edge_index, y=self.y)




class CSBM_H_MULTI:
    """
    ns: [n0, n1, ..., nn],
    mus: [mu0, mu1, ..., mun],
    sigmas: [sigma0, sigma1, ..., sigman],
    ds: [d0, d1, ..., dn]
    
    Multiclass Compatibility Matrix
    Hii = h
    Hij = (1-h) / (C-1) 
    """

    def __init__(self, ns, mus, sigmas, ds):
        
        assert all(len(lst) >= 2 for lst in [ns, mus, sigmas, ds]), \
            f"Not all received lists have at least 2 elements!"
        assert all(len(lst) == len(ns) for lst in [ns, mus, sigmas, ds]), \
            f"Not all received lists are same length!"
        assert all(len(mu) == len(mus[0]) for mu in mus), \
            f"Not all mean vectors (mu) are same length!"

        self.num_classes = len(ns)
        self.num_nodes = sum(ns)

        self.ns = ns
        self.ds = ds
        self.mus = [torch.as_tensor(mu, dtype=torch.float) for mu in mus]
        self.sigmas = sigmas
        
        self.dim = len(mus[0])
    
        self.X = None
        self.y = None

    def generate_labels(self):

        # y0 ∈ R^n0
        labels = []
        for c in range(self.num_classes):
            labels_c = torch.full((self.ns[c],), c, dtype=torch.long)
            labels.append(labels_c)

        self.y = torch.cat(labels) 
        return self.y

    def generate_features(self):
        
        # x0 ~ N(mu0, sigma0*I) ∈ R^n0xdim
        features = []
        for c in range(self.num_classes): 
            features_c = torch.distributions.MultivariateNormal(
                loc=self.mus[c], 
                covariance_matrix=self.sigmas[c]*torch.eye(self.dim),
                ).sample((self.ns[c],))
            features.append(features_c)
        
        self.X = torch.vstack(features)  
        return self.X

    def generate_compatibility_matrix(self, h):

        assert 0 <= h <= 1, "h must be between 0 and 1!"
    
        # probs balanced among all foreign classes
        H = torch.ones((self.num_classes, self.num_classes), dtype=torch.float)
        H *=  (1 - h) / (self.num_classes - 1)
        H.fill_diagonal_(h)

        return H


    def generate_edge_index(self, h, directed=False):
    
        assert self.y is not None, "Labels (y) have not been created!"
        
        H = self.generate_compatibility_matrix(h)

        class_nodes = {}
        for c in range(self.num_classes):
            class_nodes[c] = torch.where(self.y == c)[0]

        edges = []
        for u in range(self.num_nodes):

            # src node 
            y_u = int(self.y[u])
            degree_u = self.ds[y_u]

            # sample degree_u nbs
            for _ in range(degree_u):
                
                class_probs = H[y_u]
                # sample nb from class probabilities
                y_v = torch.multinomial(class_probs, num_samples=1, replacement=True).item()

                candidates = class_nodes[y_v]

                if len(candidates) == 0 or (len(candidates) == 1 and candidates[0].item() == u): 
                    continue            

                while True:
                    idx = torch.randint(len(candidates), (1, )).item()
                    v = candidates[idx] 
                    if v != u: break # no self loops

                edges.append((u, v)) 
            
                if not directed: edges.append((v, u)) 

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_index = coalesce(edge_index) 

        return edge_index


    def generate_data(self, h, directed=False):

        assert self.X is not None and self.y is not None, \
            f"Features (X) and/or labels (y) have not been created!"
        edge_index = self.generate_edge_index(h, directed)

        return Data(x=self.X, edge_index=edge_index, y=self.y)




def generate_orthogonal_mean_vectors(dim, a):
    """
    Equidistant mean vectors with dim==c (from basis vectors ∈ R^c)
    """

    mus = []
    for i in range(dim):
        mu = torch.zeros(dim)
        mu[i] = a
        mus.append(mu)

    distance = torch.norm(mus[0]-mus[-1])

    return mus


if __name__ == "__main__":

    # Test
    n = 100 
    d = 3 
    a = 1 
    sigma = 2
    c = 2 
    dim = c 

    generator = CSBM_H_MULTI(
                ns=[100,50],
                mus=[(0, 0), (3, 3)],
                sigmas=[1, 2],
                ds=[2, 5],
            )

    generator.generate_features()
    generator.generate_labels()
    data = generator.generate_data(h=0.7) 

    print(f"{data=}")
    print(f"{data.x=}")
    print(f"{data.y=}")
    print(f"{data.edge_index=}")
    print(f"{data.num_nodes=}")
    print(f"{data.num_edges=}")
    print(f"{torch.bincount(data.y)=}")
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    print(f"{deg.mean()=}")
    print(f"{deg.std()=}")
    print(f"{deg.min()=}")
    print(f"{deg.max()=}")

    H, H_counts = class_compatibility_matrix(data, c)
    print(f"H=\n{H}")
    print(f"H_counts=\n{H_counts}")
    print(f"{edge_homophily(data)=}")
    
    plot_csbm_h(data)
    x = data.x.numpy()
    y = data.y.numpy()
    if c == 2: plot_csbm_h_features(x, y)    
    else: plot_data_projection(x, y)