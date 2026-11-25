from src.utils.paths import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR, PROJECT_ROOT
from src.constants import ALL_DATASETS, LABEL_RATES_LOW
import numpy as np
from src.homophily import (
    edge_homophily,
    node_homophily, 
    class_homophily, 
    adjusted_homophily, 
    unbiased_homophily,
    preprocess_for_homophily,
    node_homophily_h2,
    rmi
)
from src.utils import (
    CSBM_H_2, 
    CSBM_H_MULTI, 
    load_config, 
    generate_splits_synthetic,
    generate_pyg_subgraph,
    generate_orthogonal_mean_vectors
    )
from src.visualization import (
    plot_graph,
    plot_loss,
    plot_accuracies,
    plot_csbm_h,
    plot_csbm_h_features,
    plot_gain_vs_homophily_syn,
    plot_gain_vs_label_rate_syn,
    plot_gain_vs_h_lr_syn,
    plot_data_projection,
    plot_measurable_homophily_scatterplots
    )
from src.homophily import edge_homophily
from src.models import GCN, MLP
from src.training import train_model
from pathlib import Path
from src.utils import set_seed
import torch
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import subprocess



# Config
config_path = CONFIGS_DIR / "base_config.yaml"
CONFIG = load_config(config_path)

# R
anova_script = PROJECT_ROOT / "src" / "evaluation" / "two_factor_rm_anova.R"

# Fixed CSBM-H params
h_vals =  np.linspace(0, 1, 11)
n = 500 
d = 5 
a = 1 
sigma = 1
C = [2, 3, 4, 5]
dim = C 

# Robustness parameters
RUNS = 100 
LABEL_RATES = LABEL_RATES_LOW 


def generate_splits_by_lr(data):

    splits_by_lr = {}

    for lr in LABEL_RATES:

        data_split = generate_splits_synthetic(
                                        data=data.clone(), 
                                        label_rate=lr, 
                                        sampling_strategy="stratified"
                                        )

        splits_by_lr[lr] = {
            "train_mask": data_split.train_mask.clone(), 
            "val_mask": data_split.val_mask.clone(),
            "test_mask": data_split.test_mask.clone(),
        }
    
    return splits_by_lr


def compute_win_rate(df_runs):

    df = df_runs.copy()
    df["gcn_wins"] = (df["gcn_test_acc"] > df["mlp_test_acc"]).astype(int)

    df_gcn_wins = df.groupby(["homophily_requested", "label_rate"])["gcn_wins"].mean()

    df_gcn_wins = df_gcn_wins.unstack("homophily_requested")

    return df_gcn_wins.sort_index(ascending=False)


def run_experiment(load_existing=True):
    
    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir = RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Repeat for different class sizes
    for c in C:

        print(f"\n\n ========== {c} Classes ==========")
        class_dir = experiment_dir / f"c{str(c)}"
        class_dir.mkdir(parents=True, exist_ok=True)

        acc_loss_dir = class_dir / "acc_loss_plots"
        acc_loss_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir = class_dir / "graphs_plots"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        anova_dir = class_dir / "anova"
        anova_dir.mkdir(parents=True, exist_ok=True)

        if load_existing:

            path_runs = class_dir / "gain_vs_homophily_runs.csv"
            df_runs = pd.read_csv(path_runs)
            print("Succesfully loaded df_runs")
            path_means = class_dir / "gain_vs_homophily.csv"
            df = pd.read_csv(path_means)
            print("Succesfully loaded df")

        else:        

            device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu")
            print(f"Using device: {device}")

            rows = []
            
            # 1. Generate RUNS datasets
            for r in range(RUNS):

                set_seed(r) # graph generation seed 
                print(f"\n\nSEED={r}")

                # Same features X and labels y, change connections h
                generator = CSBM_H_MULTI(
                    ns=[n]*c,
                    mus=generate_orthogonal_mean_vectors(c, a), 
                    sigmas=[sigma]*c,
                    ds=[d]*c,
                )
                generator.generate_features() # X
                generator.generate_labels()   # y

                splits_by_lr = generate_splits_by_lr(generator.generate_data(h=0.5))

                # 2. Generate new edge_index with h
                for h in h_vals:

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
                    if r == 0: 
                        graph_path = graphs_dir / f"graph_{dataset_name}.png"
                        features_path = graphs_dir / f"features_{dataset_name}.png"
                        plot_csbm_h(data, output_path=graph_path)
                        x = data.x.numpy()
                        y = data.y.numpy()
                        if c == 2: plot_csbm_h_features(x, y, output_path=features_path)    
                        else: plot_data_projection(x, y, output_path=features_path)
                        
                    # 3. Using same graph with certain h, evaluate models at lr
                    for lr in LABEL_RATES:
                            
                        print(f"{lr=}") 
                        data_split = data.clone() 
                        
                        data_split.train_mask = splits_by_lr[lr]["train_mask"]
                        data_split.val_mask = splits_by_lr[lr]["val_mask"]
                        data_split.test_mask = splits_by_lr[lr]["test_mask"]
                        
                        num_features = data_split.num_node_features
                        num_classes = generator.num_classes
                        labeled_nodes = data_split.train_mask.sum().item()
                        labeled_nodes_per_class =  labeled_nodes / num_classes

                        # Fit models
                        data_split = data_split.to(device)
                        training_seed = r + int(lr*1000)
                        
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

                        # Plot loss, acc
                        histories = {
                            "MLP": mlp_stats["history"],
                            "GCN": gcn_stats["history"],
                        }
                        
                        if r == 0: 
                            loss_plot_path = acc_loss_dir / f"loss_curves_{dataset_name}_lr{str(lr).replace('.','p')}.png"
                            acc_plot_path = acc_loss_dir / f"acc_curves_{dataset_name}_lr{str(lr).replace('.','p')}.png"
                            plot_loss(histories, loss_plot_path)
                            plot_accuracies(histories, acc_plot_path)

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

                            "label_rate": lr,
                            "realized_label_rate": labeled_nodes/data.num_nodes,
                            "labeled_nodes_per_class": labeled_nodes_per_class,
                            "num_classes": num_classes,   

                            "gcn_test_acc": gcn_test_acc,
                            "mlp_test_acc": mlp_test_acc,
                            "abs_gain": abs_gain,
                            "rel_gain": rel_gain,
                            "ratio": ratio,
                        }
                        rows.append(row)

            df_runs = pd.DataFrame(rows).round(5)
            output_path = class_dir / "gain_vs_homophily_runs.csv"
            df_runs.to_csv(output_path, index=False)

            df = df_runs.groupby(["homophily_requested", "label_rate"], dropna=False).agg(
                homophily_realized_mean=("homophily_realized", "mean"),
                realized_label_rate_mean=("realized_label_rate", "mean"),
                run_count=("run", "count"),

                gcn_test_acc_mean=("gcn_test_acc", "mean"),
                mlp_test_acc_mean=("mlp_test_acc", "mean"), 
                abs_gain_mean=("abs_gain", "mean"), 
                rel_gain_mean=("rel_gain", "mean"),
                ratio_mean=("ratio", "mean"),

                homophily_realized_std=("homophily_realized", "mean"),
                realized_label_rate_std=("realized_label_rate", "mean"),

                gcn_test_acc_std=("gcn_test_acc", "std"),
                mlp_test_acc_std=("mlp_test_acc", "std"), 
                abs_gain_std=("abs_gain", "std"), 
                rel_gain_std=("rel_gain", "std"),
                ratio_std=("ratio", "std"),
                ).reset_index().round(5)

            output_path = class_dir / "gain_vs_homophily.csv"
            df.to_csv(output_path, index=False)
            
        # ANOVA
        path_runs = class_dir / "gain_vs_homophily_runs.csv"
        if shutil.which("Rscript") is None:
            print("Rscript is not available in the system, skipping")
        else:
            try:
                subprocess.run(["Rscript", anova_script, path_runs, anova_dir], check=True)
            except subprocess.CalledProcessError as e:
                print(f"ANOVA script could not be executed: {e}")    

        # 2D, 3D Plots
        for y_var in ["abs_gain", "gcn_test_acc", "mlp_test_acc"]:
            output_path = class_dir / f"{y_var}_vs_homophily.png"
            plot_gain_vs_homophily_syn(df, y_var, output_path)
        y_var = "abs_gain"
        output_path = class_dir / f"{y_var}_vs_label_rate.png"
        plot_gain_vs_label_rate_syn(df, y_var, output_path)
        output_path = class_dir / f"{y_var}_vs_h_lr.png"
        plot_gain_vs_h_lr_syn(df, y_var, output_path)

    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":

    run_experiment()