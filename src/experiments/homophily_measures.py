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
from src.utils.datasets import (
    import_dataset,
    inspect_graph,
    inspect_graph_simple
)
from src.visualization import (
    plot_gain_vs_homophily,
    plot_correlation_pairgrid
)
from src.utils.paths import DATASETS_DIR, RESULTS_DIR, CONFIGS_DIR
from src.constants import ALL_DATASETS, HOMOPHILY_METRICS, DATASETS_SMALL
import pandas as pd
from pathlib import Path
import shutil
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
import numpy as np
import time



def compute_correlations(metrics, df, alpha=0.05):
    """
    Pearson, Spearman between each pairwise metrics 
    with Bonferroni correction (risk of false positives)
    """
    
    results = []

    for m1, m2 in combinations(metrics, 2):

        x = df[m1]
        y = df[m2]

        p_r, p_p = pearsonr(x, y)
        s_r, s_p = spearmanr(x, y)

        results.append({
            "metric_1": m1,
            "metric_2": m2,
            "pearson_r": round(p_r, 6),
            "pearson_p": round(p_p, 6),
            "spearman_r": round(s_r, 6),
            "spearman_p": round(s_p, 6),
        })
    
    df_corr = pd.DataFrame(results)
    
    # Bonferroni correction
    m = len(results)
    alpha_prime = alpha / m

    df_corr["alpha_prime"] = alpha_prime
    df_corr["pearson_significant"] = df_corr["pearson_p"] < alpha_prime
    df_corr["spearman_significant"] = df_corr["spearman_p"] < alpha_prime
    
    return df_corr


def run_experiment(load_existing=True):

    start = time.time()

    # Experiment dir setup
    experiment_name = Path(__file__).stem
    experiment_dir =  RESULTS_DIR / experiment_name
    if not load_existing:
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metrics_names = [m for m in HOMOPHILY_METRICS.keys()]

    if load_existing:

        path_measures = experiment_dir / "homophily_measures.csv"
        df = pd.read_csv(path_measures)
        path_correlations = experiment_dir / "homophily_correlations.csv"
        df_corr = pd.read_csv(path_correlations)

    else:
        for i, d in enumerate(ALL_DATASETS, 1):
            
            print(f"\n======================== ({i}) {d} ========================")

            dataset, graph = import_dataset(name=d, 
                            label_rate=1.0, # antes: None
                            val_ratio=0.0,
                            test_ratio=0.0,
                            root=DATASETS_DIR,
                            verbose=False)

            preprocess_for_homophily(graph)

            row = {"dataset": d}

            # Inspect homophily measures
            for name, fn in HOMOPHILY_METRICS.items():
                homophily = fn(graph)
                row[name] = homophily

            rows.append(row)

        df = pd.DataFrame(rows)

        df = df.sort_values(by="h_unb", ascending=False).reset_index(drop=True)
        output_path = experiment_dir / "homophily_measures.csv"
        df.to_csv(output_path, index=False)
        print(f"df:\n{df}")

        # Correlations
        df_corr = compute_correlations(metrics_names, df)
        output_path = experiment_dir / "homophily_correlations.csv"
        df_corr.to_csv(output_path, index=False)
        print(f"df_corr:\n{df_corr}")

    # Pairgrid
    for name in ["pearson", "spearman"]:
        file_name = f"{name}_pairgrid.png"
        plot_correlation_pairgrid(
            df=df,
            df_corr=df_corr,
            metrics=metrics_names,
            method=name,
            output_path= experiment_dir / file_name
        )
    
    end = time.time()
    print(f"Time: {end-start:.2f} [s]")


if __name__ == "__main__":

    run_experiment()