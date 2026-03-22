from typing import Dict

import numpy as np
import pandas as pd

from model import LidarCableClustering
from utils import cluster_stats, load_data, plot_clusters, plot_estimated_cable

if __name__ == "__main__":
    while True:
        difficulty = (
            input(
                "\nPlease provide a dataset difficulty to model. \nOptions: \n- easy\n- medium\n- hard\n- extrahard\n(You may append _180, _90cw, _90acw to the difficulty to apply data augmentation. e.g. easy_180)\nInput: "
            )
            .strip()
            .lower()
        )
        sample_frac = input(
            """\nPlease provide value between 0.01 & 1.0. This is the size (%) of the sample used to estimate distance between neighbours.
            (Note: As sample_frac -> 1, clustering improves but time complexity -> O(n^2))\nInput:"""
        )

        # Verify inputs
        if difficulty not in (
            "easy",
            "medium",
            "hard",
            "extrahard",
            "easy_180",
            "medium_180",
            "hard_180",
            "extrahard_180",
            "easy_90cw",
            "medium_90cw",
            "hard_90cw",
            "extrahard_90cw",
            "easy_90acw",
            "medium_90acw",
            "hard_90acw",
            "extrahard_90acw",
        ):
            print("Invalid difficulty. Enter again")
            continue
        try:
            sample_frac = float(sample_frac)
        except ValueError:
            print("Invalid sample size input. Must be between 0.01 and 1.0. Enter again")
            continue

        # Load in requested data
        lcp_data: pd.DataFrame = load_data(difficulty)

        # Perform clustering
        # min_samples = 2 as each point only needs to group with the pts adjacent to it on the cable
        model = LidarCableClustering(lcp_data, sample_fraction=sample_frac)
        lcp_data_with_labels: pd.DataFrame = model.dbscan()

        cluster_stats(pd.Series(lcp_data_with_labels["labels"]))
        plot_clusters(lcp_data_with_labels, pd.Series(lcp_data_with_labels["labels"]), difficulty, sample_frac)

        # Store returned coefs. Key: cluster, value: curve coef
        curve_coefs: Dict[int, float] = {}
        all_estimated_cables = pd.DataFrame()
        for cluster in np.unique(lcp_data_with_labels["labels"][lcp_data_with_labels["labels"] != -1]):
            subset = lcp_data_with_labels.loc[lcp_data_with_labels["labels"] == cluster].drop("labels", axis=1).reset_index(drop=True)

            # Get estimated curvature value
            (curve_coef, estimated_cable) = model.estimate_curvature_coefficient(subset)
            if curve_coef != -1 and estimated_cable is not None:
                curve_coefs[cluster] = curve_coef
                estimated_cable["label"] = cluster
                all_estimated_cables = pd.concat([all_estimated_cables, estimated_cable], ignore_index=True)
                print(f"Cluster {cluster + 1} has curvature coefficient: {curve_coef}")

        print(f"\nEstimated number of cables in LiDAR cloud points: {len(curve_coefs)}")

        plot_estimated_cable(all_estimated_cables, difficulty, sample_frac)
