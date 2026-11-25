from src.utils.datasets import load_split
from src.utils.paths import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR
from src.constants import ALL_DATASETS, DATASETS_SMALL, HOMOPHILY_METRICS, LABEL_RATES_LOW, RUNS_EMP
from src.models import GCN, MLP
from src.utils import load_config
from src.training import train_model
from src.visualization import (
    plot_gain_vs_homophily, 
    plot_accuracy_boxplots,
    plot_gains
)
from pathlib import Path
import pandas as pd
import torch
import shutil
from src.utils import set_seed
import time 


# Config
config_path = CONFIGS_DIR / "base_config.yaml"
CONFIG = load_config(config_path)


RUNS = RUNS_EMP
LABEL_RATES = LABEL_RATES_LOW + ["min"] 


def run_experiment(load_existing=True):

    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir =  RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    gains_dir = experiment_dir / "gains_plots"
    gains_dir.mkdir(parents=True, exist_ok=True)

    input_path = RESULTS_DIR / "homophily_measures" / "homophily_measures.csv"
    df_homophily = pd.read_csv(input_path)

    if load_existing == True:

        path_runs = experiment_dir / "gain_vs_homophily_runs.csv"
        df_runs = pd.read_csv(path_runs)
        path_means = experiment_dir / "gain_vs_homophily.csv"
        df = pd.read_csv(path_means)

    else: 

        device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu")
        print(f"Using device: {device}")

        rows = []

        for i, dataset_name in enumerate(ALL_DATASETS, 1):

            print(f"\n======================== ({i}) {dataset_name} ========================")
            
            prev_labeled_nodes_per_class = None

            for lr in LABEL_RATES:

                for r in range(RUNS):

                    print(f"\n--------- Label rate: {lr}, Run [{r+1}/{RUNS}] ---------")
                
                    seed = r
                    set_seed(seed)

                    # Load, download if non-existent
                    import_kwargs = {
                        "val_ratio": 0.1, # default
                        "test_ratio": 0.2, # default
                        "sampling_strategy": "random", # default
                        "verbose": False
                    }

                    dataset, data = load_split(dataset_name, 
                                               lr, 
                                               r, 
                                               **import_kwargs
                                               )
                    
                    num_features = data.num_node_features
                    num_classes = dataset.num_classes
                    labeled_nodes = data.train_mask.sum().item()
                    labeled_nodes_per_class =  labeled_nodes / num_classes

                    if r == 0:
                        if prev_labeled_nodes_per_class == labeled_nodes_per_class: 
                            print("\tskipped!")
                            break
                    
                        prev_labeled_nodes_per_class = labeled_nodes_per_class

                    # Fit models
                    data = data.to(device)

                    lr_s = lr if isinstance(lr, float) else 0
                    training_seed = r + int(lr_s)
                    set_seed(training_seed) 
                    print("\tTraining GCN")
                    gcn_stats = train_model(
                        GCN,
                        data,
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

                    training_seed = r + int(lr_s*1000)
                    set_seed(training_seed) 
                    print("\tTraining MLP")
                    mlp_stats = train_model(
                        MLP,
                        data,
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

                    mlp_test_acc = mlp_stats["best_metrics"]["test"]
                    gcn_test_acc = gcn_stats["best_metrics"]["test"]

                    # Gain definitions
                    abs_gain = gcn_test_acc - mlp_test_acc
                    rel_gain = (abs_gain / mlp_test_acc) * 100 if mlp_test_acc > 0 else None
                    ratio = gcn_test_acc / mlp_test_acc if mlp_test_acc > 0 else None

                    row = {
                        "dataset": dataset_name,
                        "run": r,
                        "label_rate": lr,
                        "realized_label_rate": int(data.train_mask.sum())/data.num_nodes,
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
        output_path = experiment_dir / "gain_vs_homophily_runs.csv"
        df_runs.to_csv(output_path, index=False)

        df_gain = df_runs.groupby(["dataset", "label_rate"], dropna=False).agg(
            realized_label_rate_mean=("realized_label_rate", "mean"),
            labeled_nodes_per_class_mean=("labeled_nodes_per_class", "mean"),

            gcn_test_acc_mean=("gcn_test_acc", "mean"),
            mlp_test_acc_mean=("mlp_test_acc", "mean"), 
            abs_gain_mean=("abs_gain", "mean"), 
            rel_gain_mean=("rel_gain", "mean"),
            ratio_mean=("ratio", "mean"),

            realized_label_rate_std=("realized_label_rate", "std"),
            labeled_nodes_per_class_std=("labeled_nodes_per_class", "std"),

            gcn_test_acc_std=("gcn_test_acc", "std"),
            mlp_test_acc_std=("mlp_test_acc", "std"), 
            abs_gain_std=("abs_gain", "std"), 
            rel_gain_std=("rel_gain", "std"),
            ratio_std=("ratio", "std"),
        ).reset_index().round(5)

        df = df_homophily.merge(df_gain, on="dataset", how="inner", validate="one_to_many")
        output_path = experiment_dir / "gain_vs_homophily.csv"
        df.to_csv(output_path, index=False)


    # Plots
    output_path = experiment_dir / "accuracy_boxplots.png"
    lr = 0.2    
    plot_accuracy_boxplots(df_runs, df_homophily, output_path, lr)

    gain = "abs_gain"
    metrics = ["h_unb"] 
    gain_vs_homophily_plot_path = experiment_dir / "gain_vs_homophily.png"
    plot_gain_vs_homophily(df=df, label_rates=LABEL_RATES, metrics=metrics, gain=gain, output_path=gain_vs_homophily_plot_path)

    for dataset_name, df_dataset in df.groupby("dataset"):
        output_path = gains_dir / f"gains_{dataset_name.lower()}.png"
        plot_gains(df_dataset, output_path)
    

    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":
    
    run_experiment()



