import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

metric_labels = {
        "h_edge": r"$h_{\text{edge}}$",
        "h_node": r"$h_{\text{node}}$",
        "h_class": r"$h_{\text{class}}$",
        "h_adj": r"$h_{\text{adj}}$",
        "h_unb": r"$h_{\text{unb}}$",
        "h_node_h2": r"$h^{2}$",
        "rmi": r"$\operatorname{RMI}$",
    }

feature_proxies_labels = {
    "generalized_edge_homophily": r"$h_{\text{GE}}$",
    "feature_assortativity_flattened": r"$\rho_{\text{flat}}$",
    "feature_assortativity_per_dimension": r"$\rho_{\text{dim}}$",
    "local_similarity_euc": r"$h_{\text{LS-euc}}$",
    "local_similarity_cos": r"$h_{\text{LS-cos}}$",
}

structure_proxies_labels = {
    "degree_assortativity": r"$\rho_{\text{deg}}$",
    "lcc_assortativity": r"$\rho_{\text{lcc}}$",
    "neighborhood_similarity": r"$h_{\text{NS}}$",
}

gain_labels = {
        "gcn_test_acc": r"$\text{Acc}_{\text{GCN}}$",
        "mlp_test_acc": r"$\text{Acc}_{\text{MLP}}$",
        "abs_gain": r"$\text{Gain}_{\text{abs}}$",
        "rel_gain": r"$\text{Gain}_{\text{rel}}$",
        "ratio":    r"$\text{Ratio}$",
    }


def get_linear_regression_function(x ,y):
     return np.polyfit(x, y, 1)



def plot_gain_vs_homophily_syn(df, gain, output_path):

    # 2D: gain = f(homophily), hue=lr 
    label_rates = sorted(df["label_rate"].unique())
    h_vals = [round(v, 1) for v in sorted(df["homophily_realized_mean"].unique())]

    plt.figure(figsize=(6, 4))

    x = "homophily_realized_mean"
    y_var = gain + "_mean"
    y_std = gain + "_std"

    cmap= plt.cm.viridis
    colors = cmap(np.linspace(0,1, len(label_rates)))

    for lr, color in zip(label_rates, colors):

        df_lr = df[df["label_rate"] == lr].sort_values("homophily_realized_mean")

        pl = plt.plot(
            df_lr[x],
            df_lr[y_var],
            #yerr=df_lr[y_std],
            #marker="x",
            #markersize=5,
            color=color,
            label=str(lr),
            linewidth=3
        )

        plt.fill_between(
            df_lr[x],
            df_lr[y_var] - df_lr[y_std],
            df_lr[y_var] + df_lr[y_std],
            alpha=0.1,
            color=color,
        )

    # label
    if y_var == "gcn_test_acc_mean":
        y_label = r"$\text{Acc}_{\text{GCN}}$"
    elif y_var == "mlp_test_acc_mean":
        y_label = r"$\text{Acc}_{\text{MLP}}$"
    elif y_var == "abs_gain_mean":
        y_label = r"$\text{Gain}_{\text{abs}}$"

    # horizontal line
    if y_var in ["abs_gain_mean", "rel_gain_mean"]: 
        y_line = 0 
    elif y_var == "ratio_mean": 
        y_line = 1
    else:  
        y_line = None

    if y_var in ["gcn_test_acc_mean", "mlp_test_acc_mean"]: 
        plt.ylim(0.52, 1.02) 

    if y_line is not None:
        plt.axhline(
            y=y_line,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            zorder=0
        )

    plt.xlabel("Homophily", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(h_vals, fontsize=13)
    plt.yticks(fontsize=14)
    if y_var == "mlp_test_acc_mean":
        pass
    else:
        plt.legend(title="Label Rate", title_fontsize=14, fontsize=13, loc="lower right",)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_gain_vs_label_rate_syn(df, gain, output_path):

    # 2D: gain = f(lr), hue=h
    h_vals = sorted(df["homophily_requested"].unique())
    lr_vals = sorted(df["label_rate"].unique())

    plt.figure(figsize=(7, 4))

    x = "label_rate"
    y_var = gain + "_mean"

    cmap= plt.cm.RdBu
    colors = cmap(np.linspace(0,1, len(h_vals)))

    for h, color in zip(h_vals, colors):

        df_h = df[df["homophily_requested"] == h].sort_values("label_rate")

        pl = plt.errorbar(
            df_h[x],
            df_h[y_var],
            #marker="o",
            #markersize=5,
            color=color,
            label=str(h),
            linewidth=3
        )

    # label
    if y_var == "gcn_test_acc_mean":
        y_label = r"$\text{Acc}_{\text{GCN}}$"
    elif y_var == "mlp_test_acc_mean":
        y_label = r"$\text{Acc}_{\text{MLP}}$"
    elif y_var == "abs_gain_mean":
        y_label = r"$\text{Gain}_{\text{abs}}$"

    # horizontal line
    if y_var in ["abs_gain_mean", "rel_gain_mean"]: 
        y_line = 0 
    elif y_var == "ratio_mean": 
        y_line = 1
    else:  
        y_line = None

    if y_line is not None:
        plt.axhline(
            y=y_line,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            zorder=0
        )


    plt.xlabel("Label Rate", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.xticks(lr_vals, fontsize=14)
    plt.yticks(fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(
        handles[::-1],
        labels[::-1],
        title="Homophily",
        title_fontsize=15,
        fontsize=13,
        bbox_to_anchor=(1.02, 1)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_gain_vs_h_lr_syn(df, gain, output_path):
    
    # 3D: gain = f(h, lr), hue=h
    xl = "homophily_requested"
    yl = "label_rate"
    zl = gain + "_mean"

    ax = plt.figure(figsize=(8,8)).add_subplot(projection="3d")
    h_vals = sorted(df[xl].unique())
    lr_vals = sorted(df[yl].unique())
    x = h_vals
    y = lr_vals
    xx, yy = np.meshgrid(x,y) # pair every x with every y

    gain_matrix = df.pivot(
        index=yl,
        columns=xl,
        values=zl
    )
    zz = gain_matrix.values

    # plane
    if zl in ["abs_gain_mean", "rel_gain_mean"]: 
        z_plane = np.zeros_like(xx)
    elif zl == "ratio_mean": 
        z_plane = np.ones_like(xx)
    else:  
        z_plane = None

    z_plane = None
    if z_plane is not None:
        ax.plot_surface(xx, yy, z_plane,
                        color="black",
                        alpha=0.4,
                        )

    cmap = ListedColormap(["darkred", "darkblue"])
    norm = BoundaryNorm([np.nanmin(zz), 0, np.nanmax(zz)], cmap.N)

    #surface
    ax.plot_surface(xx, yy, zz,
                    cmap=cmap,
                    norm=norm,
                    #edgecolor="black",
                    #linewidth=0.5,
                    alpha=0.6,
                    rstride=1,
                    cstride=1,
                    antialiased=True # smooth edges
                    )

        
    ax.view_init(azim=-60)
    
    ax.set_xticks(h_vals)
    ax.set_yticks(lr_vals)

    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="z", labelsize=16)

    ax.set_xlabel("Homophily", fontsize=24, labelpad=14)
    ax.set_ylabel("Label Rate", fontsize=24, labelpad=14)
    ax.set_zlabel(r"$\text{Gain}_{\text{abs}}$", fontsize=24, labelpad=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.5)
    # plt.show()
    plt.close()

    



def plot_gain_vs_homophily(df, label_rates, metrics, gain, output_path):

    df["label_rate"] = df["label_rate"].astype(str)
    label_rates = [str(lr) for lr in label_rates if lr != "min"]

    gain_mean = f"{gain}_mean"
    gain_std = f"{gain}_std"
    
    # dataset dots
    datasets = sorted(df["dataset"].unique()) 
    colors = plt.cm.tab20b(range(len(datasets)))
    dataset_to_color = {dataset: color for dataset, color in zip(datasets, colors)}

    # regression lines
    colors_m = plt.cm.tab20b(range(len(metrics)))
    metric_to_color = {m: color for m, color in zip(metrics, colors_m)}

    groups = { 
        "high_homophily": ["Cora", "WikiCS", "CiteSeer", "PubMed"],
        "low_homophily": ["Amazon-ratings", "Tolokers", "Chameleon", "Questions", "Squirrel", "Minesweeper", "Actor"],
        "heterophily": ["Wisconsin", "Cornell", "Roman-empire", "Texas"]
    }
    
    group_to_marker = {
        "high_homophily": "o",
        "low_homophily": "s",
        "heterophily": "^"
        } 
    
    dataset_to_marker = {}
    for group, datasets in groups.items():
        for dataset in datasets:
            dataset_to_marker[dataset] =group_to_marker[group]

    fig, axes = plt.subplots(
        len(label_rates), 
        len(metrics), 
        figsize=(len(metrics)*6, len(label_rates)*2.5),
        squeeze=False, # allow 1 metric
        sharey=True,
        ) 

    for r, lr in enumerate(label_rates):

        if lr == "min": continue
        df_lr = df[df["label_rate"] == lr] 

        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            
            for _, row in df_lr.iterrows():

                ax.errorbar(
                    row[metric], 
                    row[gain_mean],
                    yerr=row[gain_std],
                    markersize=9,
                    marker=dataset_to_marker[row["dataset"]],
                    color=dataset_to_color[row["dataset"]],
                    alpha=0.8,
                    capsize=3,
                    label=row["dataset"],
                    elinewidth=2.5,   
                    capthick=2.5,
                )

                y_line = 0 if gain in ["abs_gain", "rel_gain"] else 1
                ax.axhline(y=y_line,
                            color="grey",
                            linewidth=1,
                            linestyle="--",
                            alpha=0.8,
                            zorder=0
                            )
                
            sns.regplot(
                data=df_lr,
                x=metric,
                y=gain_mean,
                scatter=False,
                ci=95,
                ax=ax,
                line_kws={"linewidth": 1.5, "color": metric_to_color[metric], "zorder": 1, "alpha": 0.4}
            )

            ax.set_xlim(-1, 1) 
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

            if r == len(label_rates) - 1:
                ax.set_xlabel(metric_labels[metric], fontsize=22) # metric_label
            else: 
                ax.set_xlabel("") 

            if c == 0:
                ax.set_ylabel(r"$\text{Gain}_{\text{abs}}$ " + f"@{lr}", fontsize=16)
            else:
                ax.set_ylabel("")

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles, 
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Dataset",
        title_fontsize=16,
        fontsize=14,
    )

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()



def plot_correlation_pairgrid(df, df_corr, metrics, method, output_path):
    
    r_col = f"{method}_r"
    p_col = f"{method}_p"
    significant_col = f"{method}_significant"
    alpha_prime = df_corr["alpha_prime"].iloc[0]

    # for heatmap
    cmap = plt.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    # for scatterplot - coloring datasets
    datasets = sorted(df["dataset"].unique()) 
    colors = plt.cm.tab20b(range(len(datasets)))
    dataset_to_color = {dataset: color for dataset, color in zip(datasets, colors)}

    # upper - heatmap
    def annotate_correlation(x, y, **kws):
        ax = plt.gca()

        row = df_corr[
            ((df_corr["metric_1"] == x.name) & (df_corr["metric_2"] == y.name)) |
            ((df_corr["metric_2"] == x.name) & (df_corr["metric_1"] == y.name))
        ]

        r = row.iloc[0][r_col]
        p = row.iloc[0][p_col]

        significant = "*" if row.iloc[0][significant_col] else ""
        cell_text = r"$\rho$" +f" = {r:.2f}\np = {p:.4f}\n{significant}"

        ax.set_facecolor(cmap(norm(r)))
        
        ax.annotate(
            cell_text,
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16
        )

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    # lower - scatterplot w/regression line
    def scatter_with_colors(x, y, **kws):
        ax = plt.gca()

        if method == "pearson":
            # regression line
            sns.regplot(
                x=x,
                y=y,
                scatter=False,
                ci=95,
                line_kws={"linewidth": 2.0})
        else:
            sns.regplot(
                x=x,
                y=y,
                scatter=False,
                line_kws={"linewidth": 2.0},
                lowess=True, # locally weighted regression
                ci=None)
        
        sns.scatterplot(
            x=x,
            y=y,
            hue=df.loc[x.index, "dataset"], # x->dataset
            palette=dataset_to_color, # dataset->color
            s=80,
            alpha=0.8,
            ax=ax,
            legend=False
            )

    g = sns.PairGrid(df, aspect=1) # aspect ratio

    g.map_lower(scatter_with_colors)
    g.map_diag(sns.histplot)
    g.map_upper(annotate_correlation)
        
    for ax, var in zip(g.axes[-1, :], g.x_vars):
        ax.set_xlabel(metric_labels[var], fontsize=22)
        ax.tick_params(axis="x",  labelsize=13)

    for ax, var in zip(g.axes[:, 0], g.y_vars):
        ax.set_ylabel(metric_labels[var], fontsize=22)
        ax.tick_params(axis="y", labelsize=13)

    g.figure.subplots_adjust(
        left=0.08, right=0.88, top=0.98, bottom=0.15, 
        wspace=0.0, hspace=0.0,
        )
    cax = g.figure.add_axes([0.90, 0.25, 0.02, 0.5]) # [l, b, w, h]

    # color bar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = g.figure.colorbar(sm, cax=cax)
    cbar.set_label(fr"{method.capitalize()} correlation ($\rho$)", fontsize=22)
    cbar.ax.tick_params(labelsize=16)

    legend_handles = [
        mpl.lines.Line2D(
            [], [],
            marker="o", linestyle="",
            color=color, markersize=6, alpha=0.8,
            label=dataset,
        )
        for dataset, color in dataset_to_color.items()
    ]
    g.figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.0),
        ncol=min(len(datasets), 5),
        markerscale=2,
        title="Dataset",
        fontsize=18,
        title_fontsize=20,
    )

    g.figure.savefig(output_path, dpi=600)
    plt.close()


def plot_labels_bubbleplot(data, dataset_name, output_path):
    """
    Equivalent to H_counts (class-compatibility matrix)
    """

    y = data.y
    num_classes = len(torch.unique(y))

    src = data.edge_index[0]
    dst = data.edge_index[1]

    src_labels = y[src]
    dst_labels = y[dst]

    assert src_labels.size() == dst_labels.size()

    counts = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for u, v in zip(src_labels, dst_labels):
        counts[u, v] += 1 

    x, y = torch.where(counts > 0) # coordinates: (x[i], y[i]) 
    
    sizes = counts[x, y]
    sizes = 10 + (2000 * (sizes / sizes.max()))

    plt.figure(figsize=(9, 9))
    plt.scatter(y, # Destination in x-axis
                x, # Source in y-axis
                s=sizes,
                alpha=0.6)
    
    plt.gca().invert_yaxis()
    plt.xlabel("Destination node label", fontsize=16)
    plt.ylabel("Source node label", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(rf"Edge label distribution ($H^{{(counts)}}$)", fontsize=18)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_homophily_at_label_rates(df, metrics, output_path):

    # dataset dots
    datasets = sorted(df["dataset"].unique()) 
    colors_d = plt.cm.tab20b(range(len(datasets)))
    dataset2color = {dataset: color for dataset, color in zip(datasets, colors_d)}

    fig, axes = plt.subplots(
        2, 
        len(metrics), 
        figsize=(len(metrics)*10, 6*2),
        squeeze=False 
        ) 

    for c, metric in enumerate(metrics):

        df_high = df[df["label_rate"].astype(float) >= 0.2].copy() 
        df_low = df[df["label_rate"].astype(float) <= 0.2].copy() 

        for r, df_curr in enumerate([df_high, df_low]):

            ax = axes[r, c]

            # shading
            ax.axvspan(
                0,
                0.2,
                color="blue",
                alpha=0.1,
                zorder=0
            )

            for d in datasets:

                df_dataset = df_curr[df_curr["dataset"] == d].copy()

                x = df_dataset["label_rate"]
                y = df_dataset[f"{metric}_mean"]
                std = df_dataset[f"{metric}_std"]

                ax.plot(
                    x,
                    y,
                    color = dataset2color[d],
                    label=d
                )

                ax.errorbar(
                    x,
                    y,
                    yerr=std,
                    marker="o",
                    ms=4,
                    alpha=0.9,
                    linestyle=":",
                    color = dataset2color[d],
                    capsize=2
                )

                ax.fill_between(
                    x,
                    y - std,
                    y + std,
                    alpha=0.05,
                    color = dataset2color[d],
                )

            if metric in ["h_node", "h_node_h2", "h_edge", "h_class", "rmi"]: # h ∈ [0, 1]
                ax.set_ylim(-0.1, 1.1) 
            else: # h ∈ [-1, 1]
                ax.set_ylim(-1.1, 1.1) 
            ax.set_title(metric)
            sign = ">" if r == 0 else "<"
            ax.set_ylabel(f"Estimated homophily (LR {sign}= 0.2)")
            ax.set_xlabel("Label Rate")
            ax.grid(True)


    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles, 
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(len(labels), 8)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # l, b, r, t
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_measurable_homophily_scatterplots(df_runs, metrics, lr, gain, output_path):
    """
    # points per plot = (#seeds * #h levels)
    """

    df = df_runs[df_runs["label_rate"] == lr]

    cmap= plt.cm.viridis
    colors = df["homophily_requested"] # color according to h_edge
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    fig, axes = plt.subplots(
            2, 
            len(metrics),
            figsize=(3.5*len(metrics), 6),
            squeeze=False, 
            sharey=True
            )
    fig.subplots_adjust(wspace=0.1, hspace=0.4, right=0.85) 
    
    y_vals = df[gain]
    y_label = gain_labels[gain]

    for c, metric in enumerate(metrics):

        metric_label = metric_labels[metric]

        x_min = -1.05 if metric in ["h_unb", "h_adj"] else -0.05
        x_max = 1.05

        # Top row
        x_vals = df[f"{metric}@1.0"]
        p_r, p_p = spearmanr(x_vals, y_vals, nan_policy="omit")

        ax0 = axes[0, c]
        sc=ax0.scatter(x_vals, y_vals, 
                           alpha=0.7, 
                           s=40, 
                           c=colors, 
                           cmap=cmap,
                           norm=norm
                           )

        ax0.tick_params(axis="both", labelsize=12)
        ax0.set_xlabel(f"{metric_label}", fontsize=20)
        ax0.set_xlim(x_min, x_max)
        if c == 0: ax0.set_ylabel(y_label, fontsize=20)
        
        # Bottom row
        x_vals = df[f"{metric}@lr"]
        p_r, p_p = spearmanr(x_vals, y_vals, nan_policy="omit")
        
        ax1 = axes[1, c]
        ax1.scatter(x_vals, y_vals, 
                           alpha=0.7, 
                           s=40, 
                           c=colors, 
                           cmap=cmap,
                           norm=norm
                           )
        ax1.tick_params(axis="both", labelsize=12)
        ax1.set_xlabel(f"{metric_label}", fontsize=20 )
        ax1.set_xlim(x_min, x_max)
        if c == 0: ax1.set_ylabel(y_label, fontsize=20)
    
        if gain in ["abs_gain", "rel_gain"]: y_line = 0 
        elif gain == "ratio": y_line = 1
        else:  y_line = None

        if y_line is not None:
            ax0.axhline(y=y_line,
                color="grey",
                linewidth=2,
                linestyle="--",
                alpha=0.8,
                zorder=0
            )
            ax1.axhline(y=y_line,
                color="grey",
                linewidth=2,
                linestyle="--",
                alpha=0.8,
                zorder=0
            )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", pad=0.01)
    cbar.set_label(r"Requested homophily ($h$)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_homophily_estimator_diff_hist(df, metrics, gain, output_path):

    df = df.copy()

    fig, axes = plt.subplots(
            1, 
            len(metrics),
            figsize=(3.5*len(metrics), 3),
            squeeze=False, # allow 1 metric
            sharex=True,
            sharey=True
            )
    fig.subplots_adjust(wspace=0.08) 

    for c, metric in enumerate(metrics):

        metric_label = metric_labels[metric]

        ax = axes[0, c]

        diff = abs(df[f"{metric}@1.0"] - df[f"{metric}@lr"] )
        if metric in ["h_adj", "h_unb"]:
            diff /= 2 # Ndiff = diff / h_max-h_min
        
        df[f"diff_{metric}"] = diff
        
        sns.histplot(
            data=df,
            x=f"diff_{metric}",
            ax=ax,
            #bins=10,
            binwidth=0.04,
            kde=True
        )

        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel(f"{metric_label}", fontsize=24)
        ax.grid(True, zorder=0)
        if c == 0: ax.set_ylabel("Count", fontsize=22)
    
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    

def plot_gain_vs_estimators(df, label_rates, metrics, gain, output_path):

    df["label_rate"] = df["label_rate"].astype(str)
    label_rates = [str(lr) for lr in label_rates]

    gain_mean = f"{gain}_mean"
    
    # dataset dots
    datasets = sorted(df["dataset"].unique()) 
    colors_d = plt.cm.tab20b(range(len(datasets)))
    dataset_to_color = {dataset: color for dataset, color in zip(datasets, colors_d)}

    # regression lines
    colors_m = plt.cm.tab20b(range(len(metrics)))
    metric_to_color = {m: color for m, color in zip(metrics, colors_m)}

    groups = { 
        "high_homophily": ["Cora", "WikiCS", "CiteSeer", "PubMed"],
        "low_homophily": ["Amazon-ratings", "Tolokers", "Chameleon", "Questions", "Squirrel", "Minesweeper", "Actor"],
        "heterophily": ["Wisconsin", "Cornell", "Roman-empire", "Texas"]
    }
    
    group_to_marker = {
        "high_homophily": "o",
        "low_homophily": "s",
        "heterophily": "^"
        } 
    
    dataset_to_marker = {}
    for group, datasets in groups.items():
        for dataset in datasets:
            dataset_to_marker[dataset] =group_to_marker[group]

    fig, axes = plt.subplots(
        len(label_rates), 
        len(metrics), 
        figsize=(len(metrics)*5, len(label_rates)*2),
        squeeze=False
        ) 

    for r, lr in enumerate(label_rates):

        df_lr = df[df["label_rate"] == lr]

        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            
            for _, row in df_lr.iterrows():

                ax.errorbar(
                    row[metric], 
                    row[gain_mean],
                    marker=dataset_to_marker[row["dataset"]],
                    markersize=8,
                    color=dataset_to_color[row["dataset"]],
                    alpha=0.7,
                    capsize=3,
                    label=row["dataset"],
                )

            sns.regplot(
                data=df_lr,
                x=metric,
                y=gain_mean,
                scatter=False,
                ci=95,
                ax=ax,
                line_kws={"linewidth": 2.0, "color": metric_to_color[metric], "zorder": 1, "alpha": 0.5}
            )

            h, subscript = metric.split("_", 1) if "_" in metric else [metric, ""]
            metric_label = f"{h}_{{{subscript}}}"

            if r == 0:
                ax.set_title(f"Gain vs ${metric_label}$") 
            
            if r == len(label_rates) - 1:
                ax.set_xlabel(f"${metric_label}$")
            else: 
                ax.set_xlabel("") 
            
            if c == 0:
                ax.set_ylabel(f"Gain: @ LR: {lr}")
            else:
                ax.set_ylabel("")

    handles, labels = axes[0,0].get_legend_handles_labels()

    fig.legend(
        handles, 
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Dataset",
    )

    plt.tight_layout(rect=[0, 0, 0.98, 1]) # l, b, r, t
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_proxies_scatterplots(df_runs, proxies, lr, gain, output_path):
    
    df = df_runs[df_runs["label_rate"] == lr]

    cmap= plt.cm.viridis
    colors = df["homophily_requested"] # color according to h_edge
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    y_vals = df[gain]
    y_label = gain_labels[gain]

    for proxy_labels in [feature_proxies_labels, structure_proxies_labels]:

        proxies_group = [proxy for proxy in proxies if proxy in proxy_labels.keys()]

        fig, axes = plt.subplots(
                    1, 
                    len(proxies_group),
                    figsize=(4*len(proxies_group), 3),
                    squeeze=False, # allow 1 metric
                    sharey=True
                    )
        fig.subplots_adjust(wspace=0.1, right=0.85)

        for c, proxy in enumerate(proxies_group):

            x_vals = df[proxy]
            p_r, p_p = spearmanr(x_vals, y_vals, nan_policy="omit")

            ax = axes[0, c]
            sc=ax.scatter(x_vals, y_vals, 
                            alpha=0.7, 
                            s=40, 
                            c=colors, 
                            cmap=cmap,
                            norm=norm
                            )

            ax.tick_params(axis="both", labelsize=14)
            ax.set_xlabel(f"{proxy_labels[proxy]}", fontsize=20)
            
            if c == 0: ax.set_ylabel(y_label, fontsize=20)

            if gain in ["abs_gain", "rel_gain"]: y_line = 0 
            elif gain == "ratio": y_line = 1
            else:  y_line = None

            if y_line is not None:
                ax.axhline(y=y_line,
                    color="grey",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                    zorder=0
                ) 
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", pad=0.01)
        cbar.set_label(r"Requested homophily ($h$)", fontsize=14)
        cbar.ax.tick_params(labelsize=14)
    
        suffix = "_feature" if proxy_labels is feature_proxies_labels else "_structure"
        output_path_group = str(output_path).replace(".png", f"{suffix}.png")
        plt.savefig(output_path_group, dpi=600, bbox_inches="tight")
        plt.close()