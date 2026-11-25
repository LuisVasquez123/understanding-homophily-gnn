import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import torch



def get_class_colors(num_classes):
    return plt.cm.tab20b(range(num_classes))


def plot_class_distribution(class_counts, output_path):
    
    datasets = list(class_counts.keys())[::-1]
    
    fig, ax = plt.subplots(figsize=(14, 0.6*len(datasets)))

    for i, dataset_name in enumerate(datasets):

        class_counts_list = class_counts[dataset_name]
        num_classes = len(class_counts_list)
        total = sum(class_counts_list)
        left = 0

        colors = get_class_colors(num_classes)

        for j in range(num_classes):
            
            count = class_counts_list[j]
            proportion = count / total

            label = fr"$c_{{{j+1}}}$"+f"\n{str(int(count))}" if proportion >= 0.03 else ""

            ax.barh(
                dataset_name,
                proportion,
                left=left,
                color=colors[j],
                height=0.9
                #label=label
            )

            ax.text(
                left + proportion/2,
                dataset_name,
                label,
                ha="center",
                va="center",
                fontsize=14
            )

            left += proportion

    ax.set_xlim(0, 1)
    ax.set_xlabel("Class proportion", fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    for label in ax.get_yticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    #ax.set_title("Class distributions")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_graph(data, dataset_name, output_path, lcc=False, max_nodes=5000):

    G = to_networkx(data, to_undirected=True, remove_self_loops=True)

    if lcc:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    nodes = list(G.nodes())

    if len(nodes) > max_nodes:
        print(f"\t Subsampling {max_nodes} out of {len(nodes)} nodes for graph plotting.")
        nodes = nodes[:max_nodes]

    G_subgraph = G.subgraph(nodes).copy()

    num_classes = len([c for c in data.y.unique()])
    positions = nx.spring_layout(G_subgraph) 
    colors = get_class_colors(num_classes)

    node_colors = [colors[data.y[n].item()] for n in G_subgraph.nodes()]

    plt.figure(figsize=(8, 8))

    nx.draw(
        G_subgraph,
        positions, 
        node_color=node_colors,
        node_size=30,
        edge_color="lightgrey",
        arrows=False,
        linewidths=0.5,
        alpha=0.6,
        with_labels=False
    )

    #plt.title(f"Graph structure: {dataset_name.title()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_degree_distribution(data, dataset_name, output_path):
    deg_per_node = degree(data.edge_index[0], data.num_nodes).long()

    counts = torch.bincount(deg_per_node)
    degrees = torch.arange(len(counts))

    plt.figure(figsize=(8, 6))

    plt.scatter(
        degrees.numpy(), 
        counts.numpy(),
        color="purple",
        alpha=0.5,
        s=25
        )

    plt.xlabel("Degree", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    #plt.title(f"Degree distribution: {dataset_name.title()}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()




def plot_csbm_h(data, output_path=None, labeled=False):

    G = to_networkx(data, to_undirected=True)

    y = data.y.numpy()

    positions = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(5, 5))

    nx.draw_networkx_edges(
        G, 
        positions,
        alpha=0.10,
        width=1
    )

    nx.draw_networkx_nodes(
        G, 
        positions,
        node_color=y,
        cmap="coolwarm",
        node_size=200 if labeled else 80,
        alpha=0.8
    )

    if labeled:
        nx.draw_networkx_labels(
            G, 
            positions,
            labels={i: str(i) for i in G.nodes()},
            font_color="white",
            font_size=12
        )

    #plt.title("CSBM-H")
    plt.axis("off")
    plt.tight_layout()

    if output_path is not None: 
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else: 
        plt.savefig("csbmh-sample.png", dpi=300, bbox_inches="tight")
        #plt.show()