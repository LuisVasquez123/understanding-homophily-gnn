from src.utils import import_dataset
from src.utils import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR
from src.constants import ALL_DATASETS, LABEL_RATES_LOW, DATASETS_SMALL
from torch_geometric.utils import contains_self_loops, is_undirected
from src.visualization import (
    plot_class_distribution,
    plot_graph,
    plot_degree_distribution,
    plot_labels_bubbleplot
)
from src.utils import (
    inspect_graph,
    total_nodes_per_class,
    labeled_nodes_per_class,
    labeled_proportion_per_class,
    connectivity_stats,
    class_compatibility_matrix
)
from pathlib import Path
import pandas as pd

import shutil
from src.utils import set_seed
import time


LABEL_RATES = LABEL_RATES_LOW + ["min"]


def write_results(results, output_path):

    with open(output_path, "w") as f:

        for dataset_name, result in results.items():

            attributes = result["general_attributes"]
            table_lr = result["label_rate_table"]
            table_H = result["compatibility_matrix"]
            table_H_counts = result["compatibility_matrix_counts"]

            f.write(f"DATASET: {dataset_name}\n")

            f.write("(Graph)\n")
            f.write(f"\t\tNumber of graphs: {attributes['Number of graphs']}\n")
            f.write(f"\t\tNumber of node features: {attributes['Number of node features']}\n")
            f.write(f"\t\tNumber of classes: {attributes['Number of classes']}\n")
            f.write(f"\t\tHas self-loops?: {attributes['Has self-loops']}\n")
            f.write(f"\t\tIs undirected?: {attributes['Is undirected']}\n")

            f.write("(Nodes, edges)\n")
            f.write(f"\t\tNumber of nodes: {attributes['Number of nodes']}\n")
            f.write(f"\t\tNumber of edges: {attributes['Number of edges']}\n")
            f.write(f"\t\tAverage node degrees: {attributes['Average node degree']}\n")
            f.write(f"\t\tx shape: {attributes['x shape']}\n")
            f.write(f"\t\ty shape: {attributes['y shape']}\n")

            f.write("(Connectivity)\n")
            f.write(f"\t\tIs connected?: {attributes['Is connected']}\n")
            f.write(f"\t\tNumber of connected components: {attributes['Number of connected components']}\n")
            f.write(f"\t\tLargest connected component size: {attributes['Largest connected component size']}\n")
            f.write(f"\t\tHas isolated nodes?: {attributes['Has isolated nodes']}\n")
            f.write(f"\t\tNumber of isolated nodese: {attributes['Number of isolated nodes']}\n")

            line = "-" * 10*int(attributes['Number of classes']) + "\n"
            f.write("(Labels)\n")
            f.write(line)
            f.write(table_lr.to_string(index=False))
            f.write("\n")
            f.write(line)

            f.write("(Class compatibility matrix H [H2GCN] (probs))\n")
            f.write(line)
            f.write(table_H.to_string(index=True))
            f.write("\n")
            f.write(line)
            f.write("(Class compatibility matrix H [H2GCN] (counts))\n")
            f.write(line)
            f.write(table_H_counts.to_string(index=True))
            f.write("\n")
            f.write(line)
            f.write("\n")
    

def run_experiment(load_existing=True):

    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir =  RESULTS_DIR / experiment_name
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = experiment_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    degree_dist_dir = experiment_dir / "degree_distributions"
    degree_dist_dir.mkdir(parents=True, exist_ok=True)
    bubbleplots_dir = experiment_dir / "bubbleplots"
    bubbleplots_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    nodes_per_class_all = {}

    for i, dataset_name in enumerate(ALL_DATASETS, 1):

        print(f"\n------------------------ ({i}) {dataset_name} ------------------------")

        dataset, data = import_dataset(name=dataset_name, 
                        label_rate=None, 
                        root=DATASETS_DIR,
                        verbose=True)
        
        general_attributes = {
            # (Graph)
            "Number of graphs": len(dataset),
            "Number of node features": data.num_node_features,
            "Number of classes": dataset.num_classes,
            "Has self-loops": contains_self_loops(data.edge_index),
            "Is undirected": is_undirected(data.edge_index),
            # (Nodes, edges)
            "Number of nodes": data.num_nodes,
            "Number of edges": data.num_edges,
            "Average node degree": round(data.num_edges / data.num_nodes, 2),
            "x shape": data.x.shape,
            "y shape": data.y.shape
        }

        # (Connectivity)
        general_attributes.update(connectivity_stats(data)) # connectivity statistics

        # (Class compatibility matrix H [H2GCN])
        H, H_counts = class_compatibility_matrix(data, dataset.num_classes)
        cols = [f"Class {c}" for c in range(1, dataset.num_classes+1)]
        df_H = pd.DataFrame(H, columns=cols, index=cols).round(2)
        df_H_counts = pd.DataFrame(H_counts, columns=cols, index=cols)

        # (Labels)
        nodes_per_class = total_nodes_per_class(data, dataset.num_classes)
        
        rows = []
        rows.append(["Total"] + nodes_per_class)

        for lr in LABEL_RATES:

            dataset, data = import_dataset(name=dataset_name, 
                            label_rate=lr, 
                            root=DATASETS_DIR, 
                            # sampling_strategy="stratified",
                            verbose=False)
            
            lr_label = f"@{lr:.2f}" if isinstance(lr, float) else f"@{lr}"
            rows.append([lr_label] + labeled_nodes_per_class(data, dataset.num_classes))

        df_lr = pd.DataFrame(rows, columns=["   "] + cols)
        
        results[dataset_name] = {
            "general_attributes": general_attributes,
            "label_rate_table": df_lr,
            "compatibility_matrix": df_H,
            "compatibility_matrix_counts": df_H_counts,
        }
        nodes_per_class_all[dataset_name] = nodes_per_class
        
        # Graphs
        output_path = graphs_dir / f"graph_{dataset_name.lower()}.png"
        plot_graph(data, dataset_name, output_path, lcc=True)

        # Degree distributions
        output_path = degree_dist_dir / f"degree_distribution_{dataset_name.lower()}.png"
        plot_degree_distribution(data, dataset_name, output_path)

        # Bubbleplots
        output_path = bubbleplots_dir / f"label_bubbleplot_{dataset_name.lower()}.png"
        plot_labels_bubbleplot(data, dataset_name, output_path)

    output_path = experiment_dir / "eda.txt"
    write_results(results, output_path)

    output_path = experiment_dir / "class_distribution.png"
    plot_class_distribution(nodes_per_class_all, output_path)

    end = time.time()
    print(f"Time: {end-start:.2f} [s]")
    

if __name__ == "__main__":

    run_experiment()


            
        