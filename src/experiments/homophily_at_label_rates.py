from src.homophily import (
    edge_homophily,
    node_homophily, 
    class_homophily, 
    adjusted_homophily, 
    unbiased_homophily,
    preprocess_for_homophily,
    node_homophily_h2,
    rmi,
)
from src.utils import (
    import_dataset,
    inspect_graph,
    generate_pyg_subgraph,
    load_split,
)

from scipy.stats import pearsonr, spearmanr
from src.visualization import plot_homophily_at_label_rates, plot_gain_vs_estimators
from src.utils import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR
from src.constants import (
    ALL_DATASETS, 
    DATASETS_SMALL, 
    HOMOPHILY_METRICS, 
    LABEL_RATES_HIGH, 
    LABEL_RATES_LOW, 
    ALL_PROXIES, 
    RUNS_EMP)
from pathlib import Path
import shutil
import torch
import pandas as pd
from src.utils import set_seed

from joblib import Parallel, delayed
from tqdm import tqdm
import time
import numpy as np



RUNS = RUNS_EMP 
LABEL_RATES = LABEL_RATES_HIGH + LABEL_RATES_LOW


def nrmse(df, metrics):
    """
    How much do estimators of h at low label rates deviate from the ground truth
    Measure quality of predictors: RMSE = sqrt(E[(h^-h)^2])
    Normalized for measures ∈[-1,1]
    """

    df = df.copy()
    df["label_rate"] = df["label_rate"].astype(float)

    results = []
    for metric in metrics.keys():

        for dataset in df["dataset"].unique():

            df_dataset = df[df["dataset"] == dataset]
            h_true = df_dataset[df_dataset["label_rate"] == 1.0][metric].item()

            for lr in sorted(df_dataset["label_rate"].unique()):

                if lr == 1.0: 
                    continue

                h_values = df_dataset[df_dataset["label_rate"] == lr][metric]

                squared_errors = (h_values - h_true) ** 2 
                
                mse = squared_errors.mean()
                # 1. RMSE per dataset per label rate
                rmse = np.sqrt(mse)

                # normalize for measures where h∈[-1,1]
                if metric in ["h_adj", "h_unb"]:
                    rmse /= 2 

                results.append({
                    "metric": metric,
                    "dataset": dataset,
                    "label_rate": lr,
                    "nrmse": rmse,
                })
    
    df_nrmse = pd.DataFrame(results)
    
    # 2. Aggregate over datasets to get avg. RMSE for each lr 
    df_nrmse_mean = df_nrmse.groupby(["metric", "label_rate"], dropna=False).agg(
        nrmse_mean=("nrmse", "mean")
        ).reset_index().round(5) 
    
    df_nrmse_mean_wide = df_nrmse_mean.pivot(
        index="metric",
        columns="label_rate",
        values="nrmse_mean"
    )
    df_nrmse_mean_wide.columns = [f"NRMSE@{c}" for c in df_nrmse_mean_wide.columns]

    df_nrmse_mean_wide = df_nrmse_mean_wide.reset_index()
    
    return df_nrmse_mean, df_nrmse_mean_wide


def ndcg(df, metrics):
    """
    Does the homophily metric rank datasets consistently at low label rates?
    """
    def dcg(df):
        rel_list = df["rel"].fillna(0).tolist()
        return float(sum(rel / np.log2(i+1) for i, rel in enumerate(rel_list, 1)))

    df = df.copy()
    df["label_rate"] = df["label_rate"].astype(float) 

    rows = []
    for metric in metrics.keys():

        metric_name = f"{metric}_mean"
        df_metric = df[["dataset", "label_rate", metric_name]]
        
        df_gt = (
            df_metric[df_metric["label_rate"] == 1.0]
            .sort_values(by=metric_name, ascending=False)
        )
        df_gt["i_1.0"] = range(1, len(df_gt) + 1)
        df_gt["rel"] = len(df_gt) - (df_gt["i_1.0"] - 1)

        # idcg with ground truth ranking
        idcg_metric = dcg(df_gt)

        row = {
            "metric": metric,
            "iDCG": round(idcg_metric, 4),
        }

        for lr in sorted(df_metric["label_rate"].unique()):
            if float(lr) == 1.0: continue
            
            df_lr = (
                df_metric[df_metric["label_rate"] == lr]
                .dropna(subset=[metric_name]) 
                .sort_values(by=metric_name, ascending=False)
                )
            col_name = f"i_{str(lr)}"
            df_lr[col_name] = range(1, len(df_lr)+1)

            dataset2gt = df_gt.set_index("dataset")["rel"].to_dict() 
            df_lr["rel"] = df_lr["dataset"].map(dataset2gt)
            
            dcg_lr = dcg(df_lr)
            ndcg_lr = dcg_lr / idcg_metric

            row[f"nDCG@{str(lr)}"] = round(ndcg_lr, 4)
        
        rows.append(row)
    
    df_ndcg = pd.DataFrame(rows)

    return df_ndcg


def misclassification_rate(df, metrics):
    """
    How often estimator misclassified a dataset as homophilic/heterophilic (for each label rate)
    """

    df = df.copy()
    df["label_rate"] = df["label_rate"].astype(float)

    rows = []
    for metric_name in metrics.keys():

        df_metric = (
            df[["dataset", "run", "label_rate", metric_name]]
            .dropna(subset=[metric_name])
        )
        
        df_metric[metric_name] = df_metric[metric_name].astype(float)

        for dataset in df_metric["dataset"].unique():

            mask = df_metric["dataset"] == dataset
            df_dataset = df_metric[mask]

            h_true = df_dataset[df_dataset["label_rate"] == 1.0][metric_name].item()
            
            # h∈[-1,1]
            if metric_name in ["h_adj", "h_unb"]:
                
                df_metric.loc[mask, "misclassified"] = (df_metric.loc[mask, metric_name] * h_true) < 0
            # h∈[0,1]
            else:
                if h_true >= 0.5:
                    df_metric.loc[mask, "misclassified"] = df_metric.loc[mask, metric_name] < 0.5
                else: 
                    df_metric.loc[mask, "misclassified"] = df_metric.loc[mask, metric_name] >= 0.5

        row = {
            "metric": metric_name
        }
        
        # MR for each lr
        for lr in sorted(df_metric["label_rate"].unique()):
            if float(lr) == 1.0:
                continue

            df_lr = (
                df_metric[df_metric["label_rate"] == lr]
                .dropna(subset=["misclassified"])
            )

            total = df_lr["misclassified"].count()
            misclassified = df_lr["misclassified"].sum()
            
            mr = misclassified / total

            row[f"MR@{str(lr)}"] = round(mr, 4)

        rows.append(row)

    df_mr = pd.DataFrame(rows)

    return df_mr
    

def compute_homophily_measures(data):

    print(f"\tComputing homophily measures")

    preprocess_for_homophily(data)

    visible = data.train_mask.nonzero(as_tuple=True)[0]

    subgraph = generate_pyg_subgraph(data, visible_nodes=visible)
    
    homophilies = {}
    for metric_name, fn in HOMOPHILY_METRICS.items(): 

        # RMI needs full graph for commnunity detection, h2 needs full graph for N1(v)
        if metric_name == "rmi" or metric_name == "h_node_h2": 
            homophily = fn(data, visible_nodes=visible)
        # else, we construct subgraph with visible nodes
        else:
            homophily = fn(subgraph)

        homophilies[metric_name] = homophily

    return homophilies


def compute_estimator_gain_correlation(df_estimators, lr, estimators_names, gain):

    correlations = { "lr": lr }

    for h in estimators_names:

        x = df_estimators[gain]
        y = df_estimators[h]

        p_r, p_p = spearmanr(x, y)

        correlations[h] = p_r

    return correlations



def run_experiment(load_existing=True):
    
    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir =  RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)

    input_dir = RESULTS_DIR / "homophily_proxies"

    if load_existing:

        path_runs = experiment_dir / "homophily_at_label_rates_runs.csv"
        df_runs = pd.read_csv(path_runs)
        path_means = experiment_dir / "homophily_at_label_rates.csv"
        df = pd.read_csv(path_means)

    else:

        rows = []
        for i, dataset_name in enumerate(ALL_DATASETS, 1):

            print(f"\n======================== ({i}) {dataset_name} ========================")

            prev_labeled_nodes_per_class = None

            for lr in LABEL_RATES:

                for r in range(RUNS):

                    print(f"\n--------- Label rate: {lr}, Run [{r+1}/{RUNS}] ---------")

                    seed = r
                    set_seed(seed)

                    if lr in LABEL_RATES_HIGH: # were not used to train models
                        import_kwargs = {
                            "val_ratio": 0.0, 
                            "test_ratio": 0.0, 
                            "sampling_strategy": "random", # default
                            "verbose": False
                        }
                    else: # reuse splits from gain_vs_homophily.py to estimate homophily
                        import_kwargs = {
                            "val_ratio": 0.1, # default, unused
                            "test_ratio": 0.2, # default, unused
                            "sampling_strategy": "random", # default
                            "verbose": False
                        }

                    dataset, data = load_split(dataset_name, 
                                               lr, 
                                               r, 
                                               **import_kwargs
                                               )

                    num_classes = dataset.num_classes
                    labeled_nodes = data.train_mask.sum().item()
                    labeled_nodes_per_class = labeled_nodes / num_classes

                    if r == 0:
                        if prev_labeled_nodes_per_class == labeled_nodes_per_class: 
                            print("\tskipped!")
                            break

                        prev_labeled_nodes_per_class = labeled_nodes_per_class

                    homophilies_all = compute_homophily_measures(data)
                    row = {
                        "dataset": dataset_name,
                        "run": r,
                        "num_classes": num_classes,   
                        "label_rate": lr,
                        "realized_label_rate": int(data.train_mask.sum())/data.num_nodes,
                        "labeled_nodes_per_class": labeled_nodes_per_class,

                        **homophilies_all,
                        }

                    rows.append(row)
                    
                    if lr == 1.0:
                        break

        df_runs = pd.DataFrame(rows).round(5)
        output_path = experiment_dir / "homophily_at_label_rates_runs.csv"
        df_runs.to_csv(output_path, index=False)

        df = df_runs.groupby(["dataset", "label_rate"], dropna=False).agg(
            realized_label_rate_mean=("realized_label_rate", "mean"),
            labeled_nodes_per_class_mean=("labeled_nodes_per_class", "mean"),
            h_node_mean=("h_node", "mean"),
            h_edge_mean=("h_edge", "mean"),
            h_class_mean=("h_class", "mean"),
            h_adj_mean=("h_adj", "mean"),
            h_unb_mean=("h_unb", "mean"),
            h_node_h2_mean=("h_node_h2", "mean"),
            rmi_mean=("rmi", "mean"),

            realized_label_rate_std=("realized_label_rate", "std"),
            labeled_nodes_per_class_std=("labeled_nodes_per_class", "std"),
            h_node_std=("h_node", "std"),
            h_edge_std=("h_edge", "std"),
            h_class_std=("h_class", "std"),
            h_adj_std=("h_adj", "std"),
            h_unb_std=("h_unb", "std"),
            h_node_h2_std=("h_node_h2", "std"),
            rmi_std=("rmi", "std"),
        ).reset_index().round(5)

        output_path = experiment_dir / "homophily_at_label_rates.csv"
        df.to_csv(output_path, index=False)
    
    # RMSE    
    df_nrmse_mean, df_nrmse_mean_wide = nrmse(df_runs, HOMOPHILY_METRICS)
    output_path = experiment_dir / "nrmse.csv"
    df_nrmse_mean_wide.to_csv(output_path, index=False)
    output_path = experiment_dir / "nrmse_at_label_rates.png"
    plot_label_rate_rmse(df_nrmse_mean, output_path)

    # nDCG
    df_ndcg = ndcg(df, HOMOPHILY_METRICS)
    output_path = experiment_dir / "ndcg.csv"
    df_ndcg.to_csv(output_path, index=False)
    
    # Misclassification Rate
    df_mr = misclassification_rate(df_runs, HOMOPHILY_METRICS)
    output_path = experiment_dir / "mr.csv"
    df_mr.to_csv(output_path, index=False)
    


    # Plots
    # Degradation at all label rates
    output_path = experiment_dir / "homophily_at_label_rates.png"
    plot_homophily_at_label_rates(df, HOMOPHILY_METRICS, output_path)

    # Gain vs. h^ @label rates
    metric_names = [n for n in HOMOPHILY_METRICS.keys()] # true label homophily
    metric_means_names = [f"{n}_mean" for n in HOMOPHILY_METRICS.keys()] # label homphily @lr
    all_estimators_names = metric_names + ALL_PROXIES + metric_means_names
    gain = "abs_gain_mean"
    
    rows = []
    dfs_estimators_all_lr = []
    for lr in LABEL_RATES_LOW:

        df_homophily_lr = df[df["label_rate"] == lr][
            [
                "dataset",
                "label_rate",
            ] + metric_means_names
        ]
        
        lr_str = str(lr).replace('.', 'p')
        input_path = input_dir / f"gain_vs_homophily_proxies_lr{lr_str}.csv"
        df_proxies_lr = pd.read_csv(input_path)
        
        df_estimators_lr = df_proxies_lr.merge(
            df_homophily_lr,
            on=["dataset", "label_rate"],
            how="left",
            validate="one_to_one"
        ).round(4)

        output_path = experiment_dir / f"gain_vs_homophily_all_estimators_lr{lr_str}.csv"
        df_estimators_lr.to_csv(output_path, index=False)

        dfs_estimators_all_lr.append(df_estimators_lr)

        # Correlation of estimators (at this lr) with gain
        row = compute_estimator_gain_correlation(df_estimators_lr, lr, all_estimators_names, gain)
        rows.append(row)
    
    df_correlations = pd.DataFrame(rows).round(4)
    df_correlations_t = df_correlations.set_index("lr").T
    df_correlations_t = df_correlations_t.rename_axis("metric").reset_index()
    output_path = experiment_dir / "gain_estimator_correlations.csv"
    df_correlations_t.to_csv(output_path, index=False)

    df_estimators_all_lr = pd.concat(dfs_estimators_all_lr, ignore_index=True)
    df_estimators_all_lr = df_estimators_all_lr.sort_values(["dataset", "label_rate"]).reset_index(drop=True)
    output_path = experiment_dir / f"gain_vs_homophily_all_estimators_all_lr.csv"
    df_estimators_all_lr.to_csv(output_path, index=False)

    # Each measure vs. gain
    gain = "abs_gain"
    gain_vs_homophily_plot_path = experiment_dir / "gain_vs_homophily_all_estimators_all_lr.png"
    plot_gain_vs_estimators(df=df_estimators_all_lr, label_rates=LABEL_RATES_LOW, metrics=all_estimators_names, gain=gain, output_path=gain_vs_homophily_plot_path)

    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":
    
    run_experiment()