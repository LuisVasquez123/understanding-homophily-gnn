import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, single_layer=False):

        super().__init__()
        self.single_layer = single_layer
        if single_layer: # i.e. Logistic Regression
            self.linear = torch.nn.Linear(in_channels, out_channels)
        else:
            self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
            self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

    def get_embeddings(self, x, edge_index=None):

        if  self.single_layer:
            raise NotImplementedError("Not valid for single_layer")
        else:
            x = self.linear1(x) 
            x = F.relu(x)
            return x
    
    def forward(self, x, edge_index=None):

        if self.single_layer:
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.linear2(x)
        return x