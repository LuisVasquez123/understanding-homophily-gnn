import torch
from torch_geometric.datasets import (Planetoid, 
                                      WebKB, 
                                      WikipediaNetwork,
                                      WikiCS,
                                      Actor,
                                      HeterophilousGraphDataset)
from torch_geometric.transforms import (RandomNodeSplit, 
                                        Compose, 
                                        ToUndirected, 
                                        RemoveSelfLoops, 
                                        AddSelfLoops)
from torch_geometric.utils import contains_self_loops, is_undirected
import pickle
from src.utils.paths import (
    PROJECT_ROOT, 
    DATASETS_DIR, 
    RESULTS_DIR,
    SPLITS_DIR,
    )
from src.utils.splits import StratifiedNodeSplit
from src.utils.seeds import set_seed


def total_nodes_per_class(data, num_classes):
    counts = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = (data.y == c)
        counts[c] = mask.sum()

    return counts.tolist()


def labeled_nodes_per_class(data, num_classes):
    counts = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = (data.y == c) & data.train_mask
        counts[c] = mask.sum()

    return counts.tolist()


def labeled_proportion_per_class(data, num_classes):
    total = total_nodes_per_class(data, num_classes)
    labeled = labeled_nodes_per_class(data, num_classes)    

    return [round((l / t), 3) for l, t in zip(labeled, total)]


def inspect_graph_simple(data):
    print("=== Data inspection ===")
    print(f"\t Number of nodes: {data.num_nodes}")
    print(f"\t Number of edges: {data.num_edges}")
    print(f"\t x shape: {data.x.shape}")
    print(f"\t y shape: {data.y.shape}")

    if hasattr(data, "train_mask"): # if splits were specified (label_rate != None)
        print(f"\t Number of training nodes: {int(data.train_mask.sum())}") 
        print(f"\t Number of validation nodes: {int(data.val_mask.sum())}")
        print(f"\t Number of test nodes: {int(data.test_mask.sum())}")
        print(f"\t Proportion of labeled nodes: {int(data.train_mask.sum())/data.num_nodes:.5f}")


def inspect_graph(dataset, data, labels=True):
    print(f"=== Data inspection: {dataset.name} ===")

    print(f"\t Graph:")
    print(f"\t\t Number of graphs: {len(dataset)}")
    print(f"\t\t Number of node features: {dataset.num_node_features}")
    print(f"\t\t Number of classes: {dataset.num_classes}")
    print(f"\t\t Has self-loops: {contains_self_loops(data.edge_index)}")
    print(f"\t\t Is undirected?: {is_undirected(data.edge_index)}")

    print(f"\t Nodes, edges:")
    print(f"\t\t Number of nodes: {data.num_nodes}")
    print(f"\t\t Number of edges: {data.num_edges}")
    print(f"\t\t x shape: {data.x.shape}")
    print(f"\t\t y shape: {data.y.shape}")
    
    if labels: 
        print(f"\t Splits:")
        print(f"\t\t Number of training nodes: {int(data.train_mask.sum())}") 
        print(f"\t\t Number of validation nodes: {int(data.val_mask.sum())}")
        print(f"\t\t Number of test nodes: {int(data.test_mask.sum())}")

        print(f"\t Labels:")
        print(f"\t\t Avg. number of labeled nodes per class: {int(data.train_mask.sum())/dataset.num_classes}")
        print(f"\t\t (True) Proportion of labeled nodes: {int(data.train_mask.sum())/data.num_nodes:.5f}")
        print(f"\t\t Class distribution (total)   : {total_nodes_per_class(data, dataset.num_classes)}")
        print(f"\t\t Class distribution (labeled) : {labeled_nodes_per_class(data, dataset.num_classes)}")
        print(f"\t\t Labeled proportions per class: {labeled_proportion_per_class(data, dataset.num_classes)}")
    else:
        print(f"\t\t Class distribution (total)   : {total_nodes_per_class(data, dataset.num_classes)}")


def import_dataset(name,  
                   root,
                   label_rate=None,
                   val_ratio=0.1,
                   test_ratio=0.2,
                   sampling_strategy="random",
                   verbose=True):
    """
    Loads dataset with desired % of labeled nodes (training set), or "min"=1 labeled/class
    returns torch_geometric.data.data.Data object with desired label rate
    """

    # We allow label_rate = "min"
    if label_rate is not None and label_rate != "min" and not isinstance(label_rate, (int, float)):
       raise TypeError("label_rate must be an int or float.")
    if label_rate is not None and label_rate != "min" and not (0 < label_rate <= 1):
        raise ValueError("label_rate must be in (0, 1].")
    if label_rate is not None and label_rate != "min" and label_rate + val_ratio + test_ratio > 1:
        raise ValueError("label_rate + val_ratio + test_ratio must be <= 1")

    transform = Compose([
        ToUndirected(),            # add reverse edges
        RemoveSelfLoops(),
        #AddSelfLoops(),           # GCNConv will add self-loops (+avoids double-adding)!
    ])

    if name in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=str(root / "Planetoid"), 
                            name=name.lower(),
                            transform=transform)
    elif name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(root=str(root / "WebKB"), 
                        name=name.lower(),
                        transform=transform)
    elif name in ["Squirrel", "Chameleon"]: 
        dataset = WikipediaNetwork(root=str(root / "WikipediaNetwork"), 
                                   name=name.lower(),
                                   transform=transform)
    elif name == "Actor": 
        dataset = Actor(root=str(root / "Actor"),
                        transform=transform)
    elif name == "WikiCS": 
        dataset = WikiCS(root=str(root / "WikiCS"),
                         transform=transform)
    elif name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        dataset = HeterophilousGraphDataset(root=str(root / "HeterophilousGraphDataset"), 
                                            name=name.lower(),
                                            transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data = dataset[0]
    if not hasattr(dataset, "name"): dataset.name = name.lower()
    
    # Split
    if label_rate is not None:

        if isinstance(label_rate, (int, float)) and label_rate == 1.0:
            data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        else:
            total_nodes = data.num_nodes
            total_classes = dataset.num_classes

            num_val = int(val_ratio * total_nodes)
            num_test = int(test_ratio * total_nodes)

            if label_rate == "min": 
                total_training_per_class = 1
                split_transform = RandomNodeSplit( 
                    # Takes desired training nodes per class, then samples val, test globally from rest
                    split="random",
                    num_train_per_class=total_training_per_class,
                    num_val=num_val,
                    num_test=num_test
                )         
            else:
                if sampling_strategy=="stratified":
                    split_transform = StratifiedNodeSplit( 
                    # Takes desired -stratified- training nodes per class, then samples val, test globally from rest
                    train_rate_per_class=label_rate,
                    num_val=num_val,
                    num_test=num_test
                    )      
                
                else:
                    total_training = int(total_nodes * label_rate)
                    total_training_per_class = max(1, total_training // total_classes)

                    split_transform = RandomNodeSplit( 
                    # Takes desired training nodes per class, then samples val, test globally from rest
                    split="random",
                    num_train_per_class=total_training_per_class,
                    num_val=num_val,
                    num_test=num_test
                    )
            data = split_transform(data)

    if verbose: print(f"Succesfully imported {dataset}!")

    return dataset, data


def generate_splits_synthetic(data, 
                              label_rate, 
                              val_ratio=0.1,
                              test_ratio=0.2,
                              sampling_strategy="random"):
    
    if label_rate is not None and label_rate != "min" and not isinstance(label_rate, (int, float)):
       raise TypeError("label_rate must be an int or float.")
    if label_rate is not None and label_rate != "min" and not (0 < label_rate <= 1):
        raise ValueError("label_rate must be in (0, 1].")
    if label_rate is not None and label_rate != "min" and label_rate + val_ratio + test_ratio > 1:
        raise ValueError("label_rate + val_ratio + test_ratio must be <= 1")
    
    total_nodes = data.num_nodes
    total_classes = int(data.y.max().item() +1)

    num_val = int(val_ratio * total_nodes)
    num_test = int(test_ratio * total_nodes)

    # Split
    if label_rate == "min": 
        total_training_per_class = 1
        split_transform = RandomNodeSplit( 
            # Takes desired training nodes per class, then samples val, test globally from rest
            split="random",
            num_train_per_class=total_training_per_class,
            num_val=num_val,
            num_test=num_test
        )         
    else:
        if sampling_strategy=="stratified":
            split_transform = StratifiedNodeSplit( 
            # Takes desired -stratified- training nodes per class, then samples val, test globally from rest
            train_rate_per_class=label_rate,
            num_val=num_val,
            num_test=num_test
            )      
        else:
            total_training = int(total_nodes * label_rate)
            total_training_per_class = max(1, total_training // total_classes)

            split_transform = RandomNodeSplit( 
            # Takes desired training nodes per class, then samples val, test globally from rest
            split="random",
            num_train_per_class=total_training_per_class,
            num_val=num_val,
            num_test=num_test
            )

    data = split_transform(data)

    return data



def load_split(dataset_name, lr, run, **import_kwargs):
    """
    Each (dataset, lr, run) triple has one split, shared across scripts
    
    If split already exists: load and return 
    Else: generate and return it ((dataset, data))
    """

    if import_kwargs['sampling_strategy'] == "random": sampling_strategy_id = "R"     
    elif import_kwargs['sampling_strategy'] == "stratified": sampling_strategy_id = "S" 
    else: raise ValueError(f"Sampling strategy not recognized: {import_kwargs['sampling_strategy']}")

    lr_id = str(lr).replace(".", "p")
    val_id = str(import_kwargs['val_ratio']).replace('.', 'p')
    test_id = str(import_kwargs['test_ratio']).replace('.', 'p')
    
    # "dataset_lr_run__val_test_samplingstrategy"
    split_name = f"{dataset_name}_" \
                 f"lr{lr_id}_" \
                 f"r{run}__" \
                 f"val{val_id}_" \
                 f"test{test_id}_" \
                 f"ss{sampling_strategy_id}"             

    split_path = SPLITS_DIR / f"{split_name}.pkl"

    if split_path.exists():
        with open(split_path, "rb") as f:
            print(f"\tLoaded split: {split_name}")
            return pickle.load(f)
        
    print(f"\tSplit not found, genrating: {split_name}")
    set_seed(run)
    dataset, data = import_dataset(name=dataset_name, 
                                    root=DATASETS_DIR,
                                    label_rate=lr,                     
                                    **import_kwargs,
                                    )
    
    with open(split_path, "wb") as f:
        pickle.dump((dataset, data), f)

    return dataset, data
        