import os
import pandas as pd
import numpy as np
from itertools import product
from collections import Counter

from Bio import SeqIO
from Bio.Cluster import kcluster      # BioPython's built-in k-means
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def create_dir(directory_name):
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        return
    except FileNotFoundError:
        return


# ─────────────────────────────────────────────────────────────
# 1. K-MER COUNTING
# ─────────────────────────────────────────────────────────────

def count_kmers(sequence: str, k: int) -> Counter:
    """
    Slide a window of size k across a protein sequence and count
    every sub-sequence (k-mer) that appears.

    Example (k=3):
      "MVGLK" → {"MVG":1, "VGL":1, "GLK":1}
    """
    return Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))


# ─────────────────────────────────────────────────────────────
# 2. BUILD A FIXED-LENGTH FEATURE VECTOR
# ─────────────────────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"   # standard 20

def kmer_feature_vector(sequence: str, k: int) -> np.ndarray:
    """
    Convert a sequence into a normalised frequency vector whose
    length is 20^k (one slot per possible k-mer).

    Why normalise? Sequences in your CSV range from ~305-309 aa.
    Raw counts would be slightly higher for longer sequences,
    making distance metrics unfair. Dividing by total k-mers
    produced converts everything to frequencies in [0, 1].
    """
    # 1. Generate the full alphabet of all possible k-mers in
    #    alphabetical order — this is the column index for every
    #    sequence so vectors are aligned.
    all_kmers = ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]

    # 2. Count what's actually in this sequence.
    observed = count_kmers(sequence, k)

    # 3. Fill the vector: 0 for unseen k-mers, observed count otherwise.
    vec = np.array([observed.get(km, 0) for km in all_kmers], dtype=float)

    # 4. Normalise to relative frequencies.
    total = vec.sum()
    if total > 0:
        vec /= total

    return vec


# ─────────────────────────────────────────────────────────────
# 3. LOAD DATA AND BUILD THE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

def build_feature_matrix(csv_path: str, k: int = 3) -> tuple:
    """
    Read ucp_ml_dataset.csv, vectorise every sequence, and return:
      - X        : (n_seqs, 20^k) float matrix
      - labels   : list of organism names for plot annotations
      - metadata : full DataFrame for reference
    """
    df = pd.read_csv(csv_path)

    sequences = df["Sequence"].tolist()
    labels    = df["Organism"].tolist()

    print(f"Loaded {len(sequences)} sequences. Building {k}-mer vectors "
          f"(dimension = 20^{k} = {20**k})...")

    X = np.vstack([kmer_feature_vector(seq, k) for seq in sequences])
    return X, labels, df


# ─────────────────────────────────────────────────────────────
# 4. CLUSTERING
# ─────────────────────────────────────────────────────────────

def cluster_sequences(X: np.ndarray, labels: list, k_value: int,
                      method: str = "hierarchical",
                      n_clusters: int = 4,) -> np.ndarray:
    """
    Cluster the k-mer feature matrix.

    method = "hierarchical"  — agglomerative, uses Ward linkage.
                               Good default: no need to pre-specify k,
                               and the dendrogram shows how clusters merge.
    method = "kmeans"        — BioPython's kcluster (Lloyd's algorithm).
                               Fast, but you must choose n_clusters.

    Returns: cluster_ids — integer array of length n_seqs.
    """

    if method == "hierarchical":
        # Compute pairwise Euclidean distances between frequency vectors,
        # then link them using Ward's criterion (minimises within-cluster
        # variance at each merge step).
        dist_matrix  = pdist(X, metric="euclidean")
        linkage_mat  = linkage(dist_matrix, method="ward")

        # Cut the dendrogram at n_clusters groups.
        cluster_ids  = fcluster(linkage_mat, n_clusters, criterion="maxclust")

        # --- plot the dendrogram ---
        fig, ax = plt.subplots(figsize=(14, 5))
        dendrogram(linkage_mat, labels=labels, leaf_rotation=45,
                   leaf_font_size=8, ax=ax)
        ax.set_title(f"Hierarchical clustering (Ward) — k={k_value}")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.savefig(f"K{k_value}/k{k_value}_dendrogram.png", dpi=150)
        plt.show()
        print("Dendrogram saved to dendrogram.png")

    elif method == "kmeans":
        # BioPython kcluster returns (cluster_ids, error, found_optimum).
        # It runs the algorithm multiple times (npass) and returns the
        # best solution found.
        cluster_ids, error, _ = kcluster(
            X,
            nclusters=n_clusters,
            dist="e",     # Euclidean distance
            npass=10      # repeat 10 times, keep best
        )
        print(f"K-means converged. Within-cluster error: {error:.4f}")

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'hierarchical' or 'kmeans'.")

    return cluster_ids

# ─────────────────────────────────────────────────────────────
# 5. Plot the clusters
# ─────────────────────────────────────────────────────────────

def plot_clusters(X: np.ndarray, cluster_ids: np.ndarray, labels: list, title: str, k_value: int):
    """
    Reduces the high-dimensional k-mer matrix to 2D using PCA 
    and plots the clusters.
    """
    # 1. Reduce dimensions to 2D
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X)
    
    # 2. Create the plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                          c=cluster_ids, cmap='viridis', s=50, alpha=0.8)
    
    # 3. Add labels for each point (optional, can get messy if N is large)
    # for i, label in enumerate(labels):
    #     plt.annotate(label, (X_embedded[i, 0], X_embedded[i, 1]), 
    #                  fontsize=8, alpha=0.7)
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(title)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"K{k_value}/k{k_value}_cluster_scatter_plot.png", dpi=150)
    plt.show()
    print("Scatter plot saved to cluster_scatter_plot.png")


def plot_by_class(X: np.ndarray, metadata_df: pd.DataFrame, title: str, k_value: int):
    """
    Plots the PCA-reduced data colored by biological Class.
    """
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X)
    
    # Create a temporary plotting dataframe
    plot_df = pd.DataFrame(X_embedded, columns=['PCA1', 'PCA2'])
    plot_df['Class'] = metadata_df['Class'].values

    plt.figure(figsize=(12, 8))
    
    # Using Seaborn for easier categorical coloring
    sns.scatterplot(
        data=plot_df, 
        x='PCA1', y='PCA2', 
        hue='Class', 
        palette='tab20', # Good for many categories like Mammalia, Insecta, etc.
        s=60, alpha=0.9
    )
    
    plt.title(title)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Biological Class")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"K{k_value}/k{k_value}_class_distribution_plot.png", dpi=150)
    plt.show()
    
def plot_by_protein_type(X: np.ndarray, metadata_df: pd.DataFrame, title: str, k_value: int):
    # 1. Reduce dimensions
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X)
    
    # 2. Prepare Data
    plot_df = pd.DataFrame(X_embedded, columns=['PCA1', 'PCA2'])
    plot_df['Protein_Type'] = metadata_df['Protein_Type'].values

    # 3. Handle Palettes
    # Get the number of unique items to ensure we have enough colors
    num_types = plot_df['Protein_Type'].nunique()
    
    # Use 'tab20' if you have < 20 types, otherwise use 'husl' for infinite variety
    if num_types <= 20:
        current_palette = 'tab20'
    else:
        current_palette = sns.color_palette("husl", num_types)

    # 4. Create Plot
    plt.figure(figsize=(14, 9)) # Slightly larger to accommodate the legend
    sns.scatterplot(
        data=plot_df, 
        x='PCA1', y='PCA2', 
        hue='Protein_Type', 
        palette=current_palette, 
        s=80, alpha=0.8,
        edgecolor='white',
        linewidth=0.5
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    
    # Move the legend outside so it doesn't cover data points
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Protein Type")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"K{k_value}/k{k_value}_protein_type_vibrant.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────────────────────
# 6. PUTTING IT ALL TOGETHER
# ─────────────────────────────────────────────────────────────


def run_program(k_value):
    print(f"Creating plots for K{k_value}...")
    create_dir(f"K{k_value}")
    CSV     = "ucp_ml_dataset.csv"

    X, labels, df = build_feature_matrix(CSV, k=k_value)

    cluster_ids = cluster_sequences(
        X, labels,
        k_value= k_value,
        method="hierarchical",
        n_clusters=4          # try adjusting this
    )

    df["Cluster"] = cluster_ids
    # print("\nCluster assignments:")
    # print(df[["Organism", "Protein_Type", "Class", "Cluster"]]
    #        .sort_values("Cluster").to_string(index=False))

    #df.to_csv("ucp_clustered.csv", index=False)
    #print("\nSaved with cluster labels → ucp_clustered.csv")

    # Call the plotting function
    plot_clusters(X, cluster_ids, labels, f"Protein Sequence Clusters (k={k_value})", k_value)

    plot_by_class(X, df, "K-mer Distribution by Biological Class", k_value)

    plot_by_protein_type(X, df, "K-mer Analysis: Clustering by Protein Type", k_value)


if __name__ == "__main__":
    for i in range(2, 5):
        run_program(i)