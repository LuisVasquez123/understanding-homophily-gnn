from src.experiments import (
    eda,
    homophily_measures,
    gain_gcn_mlp,
    gain_vs_homophily,
    homophily_proxies,
    homophily_at_label_rates,
    sampling_strategies,
    gcn_mlp_synthetic,
    gcn_mlp_synthetic_proxies,
)
import sys
import argparse


EXPERIMENTS = {
    "eda": eda.run_experiment, 
    "homophily_measures": homophily_measures.run_experiment,
    "gain_vs_homophily": gain_vs_homophily.run_experiment,
    "homophily_proxies": homophily_proxies.run_experiment, 
    "homophily_at_label_rates": homophily_at_label_rates.run_experiment,
    "gcn_mlp_synthetic": gcn_mlp_synthetic.run_experiment,
    "gcn_mlp_synthetic_proxies": gcn_mlp_synthetic_proxies.run_experiment,
}


GREEN = "\033[1;32m" 
BLUE = "\033[1;34m" 
RED = "\033[1;31m" 
YELLOW = "\033[1;33m" 
RESET = "\033[0m" 


def print_exp_header(exp_name):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{exp_name:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiments to run from src.experiments.")

    parser.add_argument(
        "experiments",
        nargs="+",
        choices=[*EXPERIMENTS.keys(), "all"],
        help="Name(s) of the experiment(s) to run, or 'all'",
        )
    
    parser.add_argument(
        "--load-existing",
        action="store_true",
        dest="load_existing",
        help=(
            "If set, existing stats will be loaded to generate plots, "
            "else experiments will we performed from scratch"
            ),
        )

    args = parser.parse_args()

    return args


def print_experiment_runtime():
    pass



if __name__ == "__main__":
    
    args = parse_args()
    experiments = args.experiments
    load_existing = args.load_existing

    if "all" in experiments:
        queue = [e for e in EXPERIMENTS.keys()]
    else:
        queue = args.experiments
    
    for exp_name in queue:
        print_exp_header(exp_name)

        EXPERIMENTS[exp_name](load_existing)


    
    
        

