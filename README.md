# Understanding Graph Neural Network Performance in Semi-Supervised Node Classification under Label Scarcity Conditions

*Master's Thesis, M.Sc. Software Engineering and Management. Technische Universität Graz, 2026.*

**Supervisor**: Denis Helic, Assoc.Prof. Dipl.-Ing. Dr.techn. (Institute of Human-Centered Computing, TU Graz)
  
**Second Supervisor**: Hussain Hussain (Know Center Research GmbH, Graz) 

## Abstract

Graph Neural Networks (GNNs) are known to outperform graph-agnostic models in semi-supervised node classification under certain conditions. This advantage is commonly associated with the homophily phenomenon, the tendency of nodes from the same class to connect to each other. The goal of this work is to study the mechanisms that determine GNN performance and to investigate whether their advantage over graph-agnostic models can be reliably predicted under label scarcity conditions. For this, we use the Graph Convolutional Network (GCN) and the Multi-Layer Perceptron (MLP) as the canonical graph-aware and graph-agnostic architectures, respectively. We compare the accuracy of the models in synthetic as well as empirical networks, relate this performance with homophily, and experimentally test to which extent homophily estimators computed using a subset of labeled nodes in the graph are able to predict the advantage of graph-aware models before model training. We show how synthetic graphs reveal the main structural mechanism that drives GNN performance, i.e. class-level node distinguishability, the advantage of GNNs can therefore be faithfully predicted in this setting, yet the structurally heterogeneous nature of empirical datasets make the estimation, especially under label scarcity, more challenging. We provide guidelines on which label-, feature- and topology-based metrics best serve as proxies to estimate the potential advantage of graph-aware models, both in synthetic and empirical networks, and show how these can be used in label-scarce settings. Finally, we derive that model selection under label scarcity should leverage graph properties beyond label homophily, especially in empirical datasets, where model performance is driven by multiple factors. Overall, the work offers comprehensive guidelines for understanding and predicting GNN advantage under limited supervision.

## Synthetic Pipeline overview


## Empirical Pipeline overview


## Project Structure

```bash
understanding-homophily-gnn/
├── LICENSE
├── README.md
├── configs
│   └── base_config.yaml
├── figures
├── requirements.txt
└── src
    ├── constants.py
    ├── evaluation
    │   └── two_factor_rm_anova.R
    ├── experiments
    │   ├── __init__.py
    │   ├── eda.py
    │   ├── gain_vs_homophily.py
    │   ├── gcn_mlp_synthetic.py
    │   ├── gcn_mlp_synthetic_proxies.py
    │   ├── homophily_at_label_rates.py
    │   ├── homophily_measures.py
    │   └── homophily_proxies.py
    ├── homophily
    │   ├── __init__.py
    │   ├── measures.py
    │   ├── preprocessing.py
    │   └── rmi.py
    ├── main.py                        # Main entry point to run experiments
    ├── models
    │   ├── __init__.py
    │   ├── gcn.py
    │   └── mlp.py
    ├── training
    │   ├── __init__.py
    │   └── train.py
    ├── utils
    │   ├── __init__.py
    │   ├── configs.py
    │   ├── datasets.py
    │   ├── graph_utilities.py
    │   ├── paths.py
    │   ├── seeds.py
    │   ├── splits.py
    │   └── synthetic_data.py
    └── visualization
        ├── __init__.py
        ├── datasets.py
        ├── embeddings.py
        ├── homophily.py
        └── training.py
```

## Getting started

Create a virtual environment:

```bash
python3 -m venv myvenv
```

and activate:
```bash
source myvenv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```


To download the datasets for the first time, you can run the exploratory data analysis script. From the project's root:
```bash
python3 -m src.experiments.eda 
```


To run all 
```bash
python3 -m src.main all --load-existing
```

## Citation
```bibtex
@mastersthesis{vasquez2026understandinggnns,
  author  = {Luis Enrique Vásquez Prado},
  title   = {Understanding Graph Neural Network Performance in Semi-Supervised Node Classification under Label Scarcity Conditions},
  school  = {Graz University of Technology},
  address = {Graz, Austria},
  year    = {2026},
  type    = {Master's thesis},
  url     = {https://github.com/LuisVasquez123/understanding-homophily-gnn}
}
```