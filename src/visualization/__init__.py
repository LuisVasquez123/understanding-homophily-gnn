from .embeddings import plot_data_projection, plot_csbm_h_features

from .training import (
    plot_loss,
    plot_accuracies,
    plot_gains,
    plot_accuracy_boxplots,
    )

from .homophily import (
    plot_gain_vs_homophily,  
    plot_correlation_pairgrid,
    plot_labels_bubbleplot,
    plot_homophily_at_label_rates,
    plot_gain_vs_homophily_syn,
    plot_gain_vs_label_rate_syn,
    plot_gain_vs_h_lr_syn,
    plot_measurable_homophily_scatterplots,
    plot_homophily_estimator_diff_hist,
    plot_gain_vs_estimators,
    plot_proxies_scatterplots,
    )

from .datasets import(
    plot_class_distribution,
    plot_graph,
    plot_degree_distribution,
    plot_csbm_h,
)

__all__ = [
    "plot_data_projection",
    "plot_loss",
    "plot_accuracies",
    "plot_gain_vs_homophily",
    "plot_correlation_pairgrid",
    "plot_gains",
    "plot_class_distribution",
    "plot_labels_bubbleplot",
    "plot_graph",
    "plot_homophily_at_label_rates",
    "plot_accuracy_boxplots",
    "plot_csbm_h",
    "plot_csbm_h_features",
    "plot_gain_vs_homophily_syn",
    "plot_gain_vs_label_rate_syn",
    "plot_gain_vs_h_lr_syn",
    "plot_measurable_homophily_scatterplots",
    "plot_homophily_estimator_diff_hist",
    "plot_gain_vs_estimators",
    "plot_proxies_scatterplots",
]