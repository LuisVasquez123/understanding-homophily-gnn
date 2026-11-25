import torch
from torch_geometric.transforms import BaseTransform


def total_nodes_per_class(data, num_classes):
    counts = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = (data.y == c)
        counts[c] = mask.sum()

    return counts.tolist()


class StratifiedNodeSplit(BaseTransform):
    """
    Performed a node-level stratified split (train, val, test) according to the class sizes.

    Adapted from PYG: random_node_split.py
    """

    def __init__(
        self,
        train_rate_per_class,
        num_val,
        num_test,
    ):
        self.train_rate_per_class = train_rate_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = "y"

    def forward(
        self, 
        data
    ):
        y = getattr(data, self.key)
        num_nodes = data.num_nodes
        classes = y.unique()
        num_classes = len(classes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        class_counts = torch.tensor(total_nodes_per_class(data, num_classes), dtype=torch.float)
        num_train_per_class = torch.round(class_counts * self.train_rate_per_class).long()

        assert torch.all(data.y.unique() == torch.arange(num_classes)), "Classes not 0..m-1, mapping required!"

        for c in range(num_classes):
            idx_c = (y == c).nonzero(as_tuple=False).view(-1) # idx where mask is true
            idx_c = idx_c[torch.randperm(idx_c.size(0))] # shuffle
            num_train_c = max(num_train_per_class[c].item(), 1) # sample at least 1
            idx_c = idx_c[:num_train_c]
            train_mask[idx_c] = True

        # val/test sampled globally 
        remaining = (~train_mask).nonzero(as_tuple=False).view(-1)
        if self.num_val + self.num_test > remaining.size(0):
            raise ValueError("Not enough nodes to sample for val, test sets!")
        remaining = remaining[torch.randperm(remaining.size(0))]

        val_mask[remaining[:self.num_val]] = True

        test_mask[remaining[self.num_val:self.num_val + self.num_test]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        assert not (train_mask & val_mask).any()
        assert not (val_mask & test_mask).any()
        assert not (train_mask & test_mask).any()

        return data



    
