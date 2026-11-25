import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    Works on both undirected and directed graphs, as specified in edge_index
    """

    def __init__(self, in_channels, hidden_channels, out_channels, single_layer=False):

        super().__init__()

        self.single_layer = single_layer
        if single_layer:
            self.convolution = GCNConv(in_channels, out_channels)
        else:
            self.convolution1 = GCNConv(in_channels, hidden_channels)
            self.convolution2 = GCNConv(hidden_channels, out_channels)
            # normalize: bool = True -> self-loops are automatically added, 
            #   not duplicated if already existing.

    def get_embeddings(self, x, edge_index):

        if  self.single_layer:
            raise NotImplementedError("Not valid for single_layer")
        else:
            x = self.convolution1(x, edge_index)
            x = F.relu(x)
            return x
        
    def forward(self, x, edge_index):

        if self.single_layer:
            x = self.convolution(x, edge_index) # Z = A_hat.X.W0
        else:
            x = self.convolution1(x, edge_index) # H1 = relu(A_hat.X.W0)
            x = F.relu(x) 
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.convolution2(x, edge_index) # Z = H2 = A_hat.H1.W1
        return x