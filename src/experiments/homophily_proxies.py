from src.utils import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR
from src.constants import ALL_DATASETS, DATASETS_3, DATASETS_SMALL, ALL_PROXIES, LABEL_RATES_LOW, HOMOPHILY_METRICS
from src.utils import import_dataset, to_unidirectional
from src.visualization import (
    plot_labels_bubbleplot,
    plot_proxies_vs_homophily,
    plot_gain_vs_homophily
    )
from src.homophily import (
    generalized_edge_homophily,
    feature_assortativity_per_dimension,
    feature_assortativity_flattened,
    local_similarity,
    attribute_homophily,

    degree_assortativity,
    lcc_assortativity,
    label_informativeness,
    two_hop_neighbor_class_similarity,
    neighborhood_similarity,
)
from torch_geometric.utils import is_undirected, degree, to_networkx

import torch.nn.functional as F
from scipy.stats import pearsonr
import networkx as nx

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import shutil
import numpy as np
import torch
import time 


LABEL_RATES = LABEL_RATES_LOW


def run_experiment(load_existing=True):

    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir =  RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)

    input_path = RESULTS_DIR / "homophily_measures" / "homophily_measures.csv"
    df_homophily = pd.read_csv(input_path).set_index("dataset")
    input_path = RESULTS_DIR / "gain_vs_homophily" / "gain_vs_homophily.csv"
    df_gain = pd.read_csv(input_path)

    h_label = "h_unb"

    if load_existing:

        path = experiment_dir / "homophily_proxies.csv"
        df = pd.read_csv(path)

    else: 
        rows = []

        for i, dataset_name in enumerate(ALL_DATASETS, 1):

            print(f"\n======================== ({i}) {dataset_name} ========================")

            dataset, data = import_dataset(name=dataset_name, 
                                label_rate=None, 
                                root=DATASETS_DIR,
                                verbose=False)

            data.edge_index_unidirectional = to_unidirectional(data.edge_index)            

            row = {
                "dataset": dataset_name,

                # Label homophily 
                "h_adj": df_homophily.loc[dataset_name, "h_adj"].item(),
                "h_unb": df_homophily.loc[dataset_name, "h_unb"].item(),

                # Feature homophily
                "generalized_edge_homophily": generalized_edge_homophily(data)['sim_all'],
                "feature_assortativity_flattened": feature_assortativity_flattened(data), 
                "feature_assortativity_per_dimension": feature_assortativity_per_dimension(data)['r_mean_abs'],
                "local_similarity_euc": local_similarity(data, sim="euc"),
                "local_similarity_cos": local_similarity(data, sim="cos"),

                 # Structural homophily 
                "degree_assortativity": degree_assortativity(data), 
                "lcc_assortativity": lcc_assortativity(data), 
                "neighborhood_similarity": neighborhood_similarity(data),
            }

            rows.append(row)

        df = pd.DataFrame(rows).sort_values(by=h_label, ascending=False).round(5)
        output_path = experiment_dir / "homophily_proxies.csv"
        df.to_csv(output_path, index=False)

    # Gain vs. homophily proxies @all label rates
    gain = "abs_gain_mean"
    df_gain = df_gain.merge(
        df[
            ["dataset"] + ALL_PROXIES
        ],
        on="dataset",
        how="left",
        validate="many_to_one"
    )

    output_path = experiment_dir / "gain_vs_homophily_proxies_all_lr.csv"
    df_gain.to_csv(output_path, index=False)

    # Gain vs. homophily proxies @LR
    metric_names = list(HOMOPHILY_METRICS.keys())
    for lr in LABEL_RATES:
        df_gain_lr = df_gain[df_gain["label_rate"] == str(lr)][
            [
                "dataset", 
                "label_rate", 
                "gcn_test_acc_mean", 
                "mlp_test_acc_mean", 
                gain, 
            ]  + metric_names + ALL_PROXIES
            ].sort_values(by=gain, ascending=False)
        output_path = experiment_dir / f"gain_vs_homophily_proxies_lr{str(lr).replace('.', 'p')}.csv"
        df_gain_lr.to_csv(output_path, index=False)


    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":

    run_experiment()
    