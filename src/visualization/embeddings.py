import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_data_projection(x, y, train_mask=None, output_path=None, method="pca", title=None):

    if method == "t-sne":
        z = TSNE(n_components=2, init="pca", random_state=42).fit_transform(x)
    elif method == "pca":
        z = PCA(n_components=2, random_state=42).fit_transform(x)
    else:
        raise ValueError("Unrecognized dimensionality reduction method!")

    plt.figure(figsize=(4, 3))

    # rest
    plt.scatter(
        z[:, 0], 
        z[:, 1], 
        c=y, 
        cmap="coolwarm",
        s=20,
        alpha=0.8)
    
    # training
    if train_mask is not None:
        plt.scatter(
            z[train_mask, 0], 
            z[train_mask, 1], 
            c=y[train_mask], 
            edgecolors="black",
            cmap="coolwarm",
            alpha=0.8,
            s=20)
    
    label = "t-SNE" if method == "t-sne" else "PCA"
    plt.xlabel(f"{label} component 1")
    plt.ylabel(f"{label} component 2")
    if title is not None:
        plt.title(title)
    if output_path is not None: 
        plt.savefig(output_path, dpi=300)
        plt.close()
    else: plt.show()


def plot_csbm_h_features(x, y, output_path=None, title=None):
    
    if x.shape[1] != 2:
        raise ValueError("Can only plot 2d feature vectors")

    plt.figure(figsize=(6,4))

    plt.scatter(
        x[:,0],
        x[:,1],
        c=y, 
        cmap="coolwarm",
        s=100,
        alpha=0.8
    )

    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title("CSBM-H feature space", fontsize=16)
    if output_path is not None: 
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else: 
        plt.savefig("csbmh-sample_feat.png", dpi=300, bbox_inches="tight",  pad_inches=0.5)
        #plt.show()