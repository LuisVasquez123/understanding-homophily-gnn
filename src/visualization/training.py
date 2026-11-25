import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

colors = {
    "GCN": "darkblue",
    "MLP": "darkred"
}

def plot_loss(histories, output_path):

    plt.figure(figsize=(7, 5))

    for model_name, history in histories.items():
    
        train_loss_history = history['train_loss_history']
        val_loss_history = history['val_loss_history']
        epochs = range(1, len(train_loss_history)+1)
        
        line, = plt.plot(
            epochs,
            train_loss_history,
            label=f"{model_name} (train)",
            color=colors[model_name],
            alpha=1
        )
        plt.plot(
            epochs,
            val_loss_history,
            label=f"{model_name} (val)",
            color=line.get_color(),
            alpha=0.3
        )
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(-0.05, 0.9)
    plt.title("Training/Validation loss over epochs")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    

def plot_accuracies(histories, output_path):
    plt.figure(figsize=(7, 5))

    for model_name, history in histories.items():
    
        for acc_name in ["train_acc_history", "val_acc_history", "test_acc_history"]:
            
            acc_history = history[acc_name]
            
            epochs = range(1, len(acc_history)+1)

            plt.plot(
                epochs,
                acc_history,
                label=f"{model_name}: {acc_name.replace('_acc_history', '').title()}"
            )
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    
def plot_gains(df, output_path):
    
    df = df.sort_values(by="realized_label_rate_mean")
    x = df["realized_label_rate_mean"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax2 = ax1.twinx()
    
    ax1.set_zorder(1)
    ax2.set_zorder(0)

    ax1.patch.set_visible(False)

    ax1_marker = "o"
    ax1_markersize = 8
    ax1_alpha = 0.9
    ax1_linewidth = 2
    ax1_elinewidth = 1
    ax1_capsize = 2

    ax2_marker = "x"
    ax2_markersize = 10
    ax2_alpha = 0.9
    ax2_linewidth = 2
    ax2_elinewidth = 1
    ax2_capsize = 2

    # Acc
    gcn_plot = ax1.errorbar(
        x,
        y=df["gcn_test_acc_mean"],
        #yerr=df["gcn_test_acc_std"],
        marker=ax1_marker,
        markersize=ax1_markersize,
        alpha=ax1_alpha,
        label="GCN",
        linewidth=ax1_linewidth,
        elinewidth=ax1_elinewidth,
        color="darkblue",
        capsize=ax1_capsize,
    )

    mlp_plot = ax1.errorbar(
        x,
        y=df["mlp_test_acc_mean"],
        #yerr=df["mlp_test_acc_std"],
        marker=ax1_marker,
        markersize=ax1_markersize,
        alpha=ax1_alpha,
        label="MLP",
        linewidth=ax1_linewidth,
        elinewidth=ax1_elinewidth,
        color="darkred",
        capsize=ax1_capsize,
    )

    ax1.fill_between(
        x,
        df["gcn_test_acc_mean"] - df["gcn_test_acc_std"],
        df["gcn_test_acc_mean"] + df["gcn_test_acc_std"],
        alpha=0.05,
        color="darkblue"
    )

    ax1.fill_between(
        x,
        df["mlp_test_acc_mean"] - df["mlp_test_acc_std"],
        df["mlp_test_acc_mean"] + df["mlp_test_acc_std"],
        alpha=0.05,
        color="darkred"
    )

    # abs 
    gain_plot = ax2.errorbar(
        x,
        y=df["abs_gain_mean"],
        #yerr=df["abs_gain_std"],
        marker=ax2_marker,
        markersize=ax2_markersize,
        linewidth=ax2_linewidth,
        elinewidth=ax2_elinewidth,
        alpha=ax2_alpha,
        color="gray",
        capsize=ax2_capsize,
        label=r"$\text{Gain}_{\text{abs}}$",
    )

    ax2.fill_between(
        x,
        df["abs_gain_mean"] - df["abs_gain_std"],
        df["abs_gain_mean"] + df["abs_gain_std"],
        alpha=0.1,
        color="gray"
    )

    ax2.axhline(
        y=0,
        color="grey",
        linewidth=2,
        linestyle="--",
        alpha=0.6
    )

    ax1.set_xlabel("Label rate", fontsize=20)
    ax1.set_ylabel("Test Accuracy", fontsize=20)
    ax2.set_ylabel(r"$\text{Gain}_{\text{abs}}$", fontsize=20, color="gray")

    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=16)
    ax2.tick_params(axis="y", labelsize=16, colors="gray")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax1.grid(which="major", alpha=0.3)
    
    handles = [gcn_plot, mlp_plot, gain_plot]
    labels = ["GCN", "MLP", r"$\text{Gain}_{\text{abs}}$"]

    ax1.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=16, 
        ncol=3
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def plot_accuracy_boxplots(df_runs, df_homophily, output_path, lr=0.2):

    df_runs = df_runs[df_runs["label_rate"] == str(lr)].copy()
    df_homophily = df_homophily[["dataset", "h_unb"]]
    df_homophily = df_homophily.sort_values(by="h_unb", ascending=True)

    df = df_runs.merge(df_homophily[["dataset", "h_unb"]], on="dataset", how="left")
    df = df.rename(columns={
        "gcn_test_acc": "GCN",
        "mlp_test_acc": "MLP"
    })
    
    ordering = df_homophily["dataset"].tolist()

    df = pd.melt(
        df,
        id_vars=["dataset", "h_unb", "run"],
        value_vars=["GCN", "MLP"],
        var_name="Model",
        value_name="accuracy"
    )

    plt.figure(figsize=(12, 5))
    
    ax = sns.boxplot(
        data=df,
        x="dataset",
        y="accuracy",
        hue="Model",
        order=ordering,
        gap=0.05,
        showmeans=True,
        meanprops={
            "marker": "x",
            "markeredgecolor": "black",
            
            "markersize": 4
        },
        boxprops={"alpha": 0.7},
        palette=colors
    )

    idx = [ordering.index("Wisconsin"), ordering.index("Amazon-ratings")]
    for i in idx:
        ax.axvline(
            x=i+0.5,
            color="lightgray",
            linestyle="--",
            alpha=0.8,
            linewidth=2
        )
    
    #ax.set_title(f"Test Accuracy distributions (label rate: {str(lr)})")
    ax.set_xlabel("Dataset", fontsize=16)
    ax.set_ylabel("Test Accuracy", fontsize=16)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_rotation(30)
        label.set_ha("right")

    handles, labels = ax.get_legend_handles_labels()
    mean_handle = Line2D([], [],
        marker="x",
        linestyle="None",
        markeredgecolor="black",
        markersize=6,
        label="Mean",
    )
    ax.legend(
        handles=handles + [mean_handle],
        labels=labels + ["Mean"],
        title="Model",
        fontsize=14,
        title_fontsize=16,
        loc="lower right",
    )
   
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()