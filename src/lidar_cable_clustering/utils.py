from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data(difficulty: str) -> pd.DataFrame:
    """
    Reads in requested dataset from /data.
    Args:
        difficulty: str - difficulty of the data set to model.
    Returns:
        pd.DataFrame - Requested dataset.
    Raises:
        ValueError: If the provided difficulty is not a valid option.
    """
    base_difficulties = ("easy", "medium", "hard", "extrahard")
    augmentation_exts = ("_180", "_90cw", "_90acw")

    if difficulty in base_difficulties:
        return pd.read_parquet(DATA_DIR / f"lidar_cable_points_{difficulty}.parquet").reset_index(drop=True)

    for suffix in augmentation_exts:
        if difficulty.endswith(suffix) and difficulty[: -len(suffix)] in base_difficulties:
            return pd.read_parquet(DATA_DIR / f"synthetic_data/lidar_cable_points_{difficulty}.parquet").reset_index(drop=True)

    raise ValueError(f"Invalid difficulty provided: '{difficulty}'")


def cluster_stats(labels: pd.Series):
    """
    Prints statistics on the clusters in the dataset.
    Args:
        labels: pd.Series - The labels of the clusters
    """
    print(f"\nNumber of clusters (PRIOR TO FILTERING): {len(labels.loc[labels != -1].unique())}")
    print(f"Number of noise points: {np.sum(labels == -1)}")
    print("Number of points in each cluster:")
    [
        print(f"Cluster: {cluster_id + 1}: {count}")
        for (cluster_id, count) in zip(range(0, len(labels[labels != -1])), list(np.bincount(labels[labels != -1])))
    ]


def plot_clusters(lcp_data: pd.DataFrame, labels: pd.Series, difficulty: str | None = None, sample_frac: float | None = None):
    """
    Plots the clusters in a 3D plot
    Args:
        lcp_data: pd.DataFrame - The data to plot the clusters of
        labels: pd.Series - The labels of the clusters
        difficulty: str | None - Dataset difficulty label for saving
        sample_frac: float | None - Sample fraction used for saving
    """
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Unique cluster labels
    unique_labels = set(labels[labels != -1])
    clusters = {}
    for k in unique_labels:
        mask = labels == k
        pts = lcp_data[mask]
        clusters[k] = pts
        if k == -1:
            ax.scatter(pts.x, pts.y, pts.z, c="k", marker="x", label="noise", s=1)
        else:
            ax.scatter(pts.x, pts.y, pts.z, label=f"cluster {k}", s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if difficulty and sample_frac:
        plt.savefig(RESULTS_DIR / f"clusters/{difficulty}_clusters_with_{sample_frac}_sample.png")
    ax.legend()
    ax.set_title(f"Clusters for {difficulty} with {sample_frac} sample")

    plt.show()


def plot_estimated_cable(estimated_cables: pd.DataFrame, difficulty: str | None = None, sample_frac: float | None = None):
    """
    Plots the estimated cable catenary curves in a 3D plot.
    Args:
        estimated_cables: pd.DataFrame - Combined estimated cable points with columns x, y, z, label
        difficulty: str | None - Dataset difficulty label for saving
        sample_frac: float | None - Sample fraction used for saving
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for label in estimated_cables["label"].unique():
        cable = estimated_cables[estimated_cables["label"] == label]
        ax.scatter(cable["x"], cable["y"], cable["z"], label=f"Cable {label}", s=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if difficulty and sample_frac:
        plt.savefig(RESULTS_DIR / f"identified_cables/{difficulty}_estimated_with_{sample_frac}_sample.png")
    ax.legend()
    ax.set_title(f"Estimated cables for {difficulty} with {sample_frac} sample")
    plt.show()
