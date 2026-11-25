from src.homophily.measures import *
from src.homophily.rmi import rmi

# Datasets
ALL_DATASETS = ["Cora", "CiteSeer", "PubMed",
                "Cornell", "Texas", "Wisconsin", 
                "Squirrel", "Chameleon", 
                "Actor", "WikiCS", 
                "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]
DATASETS_3 = ["Cora", "Roman-empire", "Texas"] # homo, mid, hetero
DATASETS_SMALL = ["Cora", "CiteSeer", # for quick testing
                "Cornell", "Texas", "Wisconsin"]

# Homophily
HOMOPHILY_METRICS = {
    "h_node": node_homophily,
    "h_edge": edge_homophily,
    "h_class": class_homophily,
    "h_adj": adjusted_homophily, 
    "h_unb": unbiased_homophily,
    "h_node_h2": node_homophily_h2,
    "rmi": rmi,
    }
HOMOPHILY_FEATURE_PROXIES = [
    "generalized_edge_homophily",
    "feature_assortativity_flattened",
    "feature_assortativity_per_dimension",
    "local_similarity_euc",
    "local_similarity_cos",
    ]
HOMOPHILY_STRUCTURE_PROXIES = [
    "degree_assortativity",
    "lcc_assortativity",
    "neighborhood_similarity",
]
ALL_PROXIES = HOMOPHILY_FEATURE_PROXIES + HOMOPHILY_STRUCTURE_PROXIES

# Label rates
LABEL_RATES_HIGH = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] 
LABEL_RATES_LOW = [0.20, 0.15, 0.10, 0.05, 0.01]
LABEL_RATES_LOW_FINE = [0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05, 0.025, 0.01]
LABEL_RATES = LABEL_RATES_HIGH + LABEL_RATES_LOW

# Runs
RUNS_EMP = 5