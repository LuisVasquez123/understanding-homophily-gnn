from src.utils.paths import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR, PROJECT_ROOT
from src.constants import ALL_DATASETS, HOMOPHILY_METRICS, ALL_PROXIES
import numpy as np
from src.homophily import (
    preprocess_for_homophily,
    edge_homophily,
    node_homophily, 
    class_homophily, 
    adjusted_homophily, 
    unbiased_homophily,    
    node_homophily_h2,
    rmi,
    generalized_edge_homophily,
    feature_assortativity_per_dimension,
    feature_assortativity_flattened,
    local_similarity,
    degree_assortativity,
    lcc_assortativity,
    neighborhood_similarity,
)
from src.utils import (
    CSBM_H_MULTI, 
    load_config, 
    generate_splits_synthetic,
    generate_pyg_subgraph,
    generate_orthogonal_mean_vectors,
    to_unidirectional
    )
from src.visualization import (
    plot_csbm_h,
    plot_measurable_homophily_scatterplots,
    plot_homophily_estimator_diff_hist,
    plot_proxies_scatterplots
    )
from src.models import GCN, MLP
from src.training import train_model
from pathlib import Path
from src.utils import set_seed
import torch
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import random


# Config
config_path = CONFIGS_DIR / "base_config.yaml"
CONFIG = load_config(config_path)

# Fixed CSBM-H params
n = 500 
d = 5 
a = 1 
sigma = 1
c = 2 
dim = c 

# Robustness parameters
RUNS = 500
LR = 0.05


def compute_homophily_measures_syn(data, dataset_name, run, lr, graphs_dir):

    print(f"\tComputing homophily measures")

    preprocess_for_homophily(data)

    visible = data.train_mask.nonzero(as_tuple=True)[0]
    
    # If lr=1.0, subgraph is same as graph
    subgraph = generate_pyg_subgraph(data, visible_nodes=visible)
    
    if lr != 1.0:
        lr_string = f"{lr:.2f}".replace('.','p')
        output_path = graphs_dir / f"subgraph_{lr_string}_{dataset_name}.png"
        if run < 10: plot_csbm_h(subgraph, output_path)
            
    id = "1.0" if lr == 1.0 else "lr"
    homophilies = {}
    for metric_name, fn in HOMOPHILY_METRICS.items():

        # RMI needs full graph for commnunity detection, h2 needs full graph for N1(v)
        if metric_name == "rmi" or metric_name == "h_node_h2": 
            homophily = fn(data, visible_nodes=visible)
        # else, we construct subgraph with visible nodes
        else:
            homophily = fn(subgraph)

        print(f"\t\t{metric_name}@{str(lr)}={homophily:.4f}")
        homophilies[f"{metric_name}@{id}"] = homophily
    
    return homophilies


def compute_homophily_proxies_syn(data):

    data.edge_index_unidirectional = to_unidirectional(data.edge_index)

    proxies = {
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
    
    return proxies


def nrmse(df, metrics):
    """
    How much do estimators of h at low label rates deviate from the ground truth
    Measure quality of predictors: RMSE = sqrt(E[(h^-h)^2])
    Normalized for measures ∈[-1,1]
    """

    df_all = df.copy()
    df_homo = df[df["homophily_requested"] >= 0.5]
    df_hetero = df[df["homophily_requested"] < 0.5]

    results = []

    for metric in metrics:

        row = { "metric": metric }

        for df in [df_all, df_homo, df_hetero]:
            
            h_true = df[f"{metric}@1.0"]
            h_values = df[f"{metric}@lr"]

            squared_errors = (h_values - h_true) ** 2
            mse = squared_errors.mean()
            rmse = np.sqrt(mse)

            if metric in ["h_adj", "h_unb"]:
                rmse /= 2 

            if df is df_all: nrmse_label = "all"
            elif df is df_homo: nrmse_label = "homo"
            else: nrmse_label = "hetero"
            
            row[f"nrmse_{nrmse_label}"] = rmse
        
        results.append(row)

    df_nrmse = pd.DataFrame(results).round(4)

    return df_nrmse



def run_experiment(load_existing=True):

    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir = RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = experiment_dir / "graphs_plots"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if load_existing:

        path_runs = experiment_dir / "gain_vs_homophily_runs.csv"
        df_runs = pd.read_csv(path_runs)
        print("Succesfully loaded df_runs")

    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        print(f"Using device: {device}")

        rows = []

        for r in range(RUNS):

            set_seed(r) # graph generation seed 
            print(f"\n\nSEED={r}")

            generator = CSBM_H_MULTI(
                ns=[n]*c,
                mus=generate_orthogonal_mean_vectors(dim, a), 
                sigmas=[sigma]*c,
                ds=[d]*c,
            )
            generator.generate_features() # X
            generator.generate_labels()   # y

            h = round(random.uniform(0, 1), 2)
            print(f"{h=}")

            data = generator.generate_data(h) 
            
            dataset_name = (
                    f"csbm_h__"
                    f"s{r}_"
                    f"c{c}_"
                    f"n{n}_"
                    f"d{d}_"
                    f"a{a}_"
                    f"sigma{sigma}_"
                    f"h{h:.2f}".replace('.','p')
                )
            print(f"{dataset_name=}")
            
            output_path = graphs_dir / f"graph_{dataset_name}.png"    
            if r<10: plot_csbm_h(data, output_path)
                
            # Homophily measures @lr=1.0
            data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool) 
            homophilies_full = compute_homophily_measures_syn(data, 
                                                          dataset_name, 
                                                          run=r, 
                                                          lr=1.0,
                                                          graphs_dir=graphs_dir)
            
            # Proxies
            proxies = compute_homophily_proxies_syn(data)
                    
            data_split = generate_splits_synthetic(
                                data=data.clone(), 
                                label_rate=LR, 
                                sampling_strategy="stratified"
                                )
            
            num_features = data_split.num_node_features
            num_classes = generator.num_classes
            labeled_nodes = data_split.train_mask.sum().item()
            labeled_nodes_per_class =  labeled_nodes / num_classes

            # Homophily measures @lr=lr
            homophilies_lr = compute_homophily_measures_syn(data_split, 
                                                        dataset_name,
                                                        run=r,
                                                        lr=LR,
                                                        graphs_dir=graphs_dir)

            # Fit models
            data_split = data_split.to(device)
            training_seed = r + int(LR*1000)
            
            set_seed(training_seed) 
            print("\tTraining GCN")
            gcn_stats = train_model(
                GCN,
                data_split,
                num_features=num_features,
                hidden_channels=CONFIG["models"]["hidden_channels"],
                num_classes=num_classes,
                device=device,
                epochs=CONFIG["training"]["epochs"],
                early_stopping=CONFIG["training"]["early_stopping"],
                patience=CONFIG["training"]["patience"],
                optimizer_name=CONFIG["training"]["optimizer"],
                lr=CONFIG["training"]["learning_rate"],
                weight_decay=CONFIG["training"]["weight_decay"],
                verbose=False,
            )

            set_seed(training_seed)
            print("\tTraining MLP")
            mlp_stats = train_model(
                MLP,
                data_split,
                num_features=data_split.num_features,
                hidden_channels=CONFIG["models"]["hidden_channels"],
                num_classes=num_classes,
                device=device,
                epochs=CONFIG["training"]["epochs"],
                early_stopping=CONFIG["training"]["early_stopping"],
                patience=CONFIG["training"]["patience"],
                optimizer_name=CONFIG["training"]["optimizer"],
                lr=CONFIG["training"]["learning_rate"],
                weight_decay=CONFIG["training"]["weight_decay"],
                verbose=False,
            )

            histories = {
                "MLP": mlp_stats["history"],
                "GCN": gcn_stats["history"],
            }

            mlp_test_acc = mlp_stats["best_metrics"]["test"]
            gcn_test_acc = gcn_stats["best_metrics"]["test"]

            # Gain definitions
            abs_gain = gcn_test_acc - mlp_test_acc
            rel_gain = (abs_gain / mlp_test_acc) * 100 if mlp_test_acc > 0 else None
            ratio = gcn_test_acc / mlp_test_acc if mlp_test_acc > 0 else None

            row = {
                "dataset": dataset_name,
                "run": r,
                "homophily_requested": h,
                "homophily_realized": edge_homophily(data),

                "label_rate": LR,
                "realized_label_rate": labeled_nodes/data.num_nodes,
                "labeled_nodes_per_class": labeled_nodes_per_class,
                "num_classes": num_classes,   

                **homophilies_full,
                **homophilies_lr,
                **proxies,

                "gcn_test_acc": gcn_test_acc,
                "mlp_test_acc": mlp_test_acc,
                "abs_gain": abs_gain,
                "rel_gain": rel_gain,
                "ratio": ratio,
        
            }
            rows.append(row)

        df_runs = pd.DataFrame(rows).round(5)
        output_path = experiment_dir / "gain_vs_homophily_runs.csv"
        df_runs.to_csv(output_path, index=False)

    metrics_names = [n for n in HOMOPHILY_METRICS.keys()] # true label homophily
    metric_lr_names = [f"{n}@lr" for n in HOMOPHILY_METRICS.keys()]# label homphily @lr
    all_estimators_names = metrics_names + ALL_PROXIES + metric_lr_names

    # NRMSE
    df_nrmse = nrmse(df_runs, metrics_names) 
    output_path = experiment_dir / "nrmse.csv"
    df_nrmse.to_csv(output_path, index=False)
    output_path = experiment_dir / "nrmse.png"
    plot_nrmse_syn(df_nrmse, output_path)

    # Plots
    y_var = "abs_gain"
    df_homo = df_runs[df_runs["homophily_requested"] >= 0.5]
    df_hetero = df_runs[df_runs["homophily_requested"] < 0.5]
    for df in [df_runs, df_homo, df_hetero]:
        if df is df_runs: label = "all"
        elif df is df_homo: label = "homo"
        else: label = "hetero"
        output_path = experiment_dir / f"measurable_homophily_scatterplot_{label}.png"
        plot_measurable_homophily_scatterplots(df, metrics_names, LR, y_var, output_path)
        output_path = experiment_dir / f"proxies_scatterplot_{label}.png"
        plot_proxies_scatterplots(df, ALL_PROXIES, LR, y_var, output_path) 

    output_path = experiment_dir / f"estimator_differences_histogram.png"
    plot_homophily_estimator_diff_hist(df_runs, metrics_names, y_var, output_path)

    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":

    run_experiment()        
    