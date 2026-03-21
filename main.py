import numpy as np
import pandas as pd

from model import LidarCableClustering
from utils import cluster_stats, load_data, plot_clusters

if __name__ == "__main__":
    while True:
        difficulty = input("\nPlease provide a dataset diffuclty to model. \nOptions: \n- easy\n- medium\n- hard\n- extrahard\nInput: ")
        sample_frac = input(
            """\nPlease provide value between 0.01 & 1.0. This is the size (%) of the sample used to estimate distance between neighbours.
            (Note: As sample_frac -> 1, clustering improves but time complexity -> O(n^2))\nInput:"""
        )

        # Verify inputs
        if difficulty.lower() not in ("easy", "medium", "hard", "extrahard"):
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
        lcp_data_with_labels: pd.DataFrame = LidarCableClustering.dbscan(lcp_data, min_samples=2, sample_frac=sample_frac)

        cluster_stats(pd.Series(lcp_data_with_labels["labels"]))
        plot_clusters(lcp_data_with_labels, pd.Series(lcp_data_with_labels["labels"]))

        curve_coefs = []
        for cluster in np.unique(lcp_data_with_labels["labels"]):
            subset = lcp_data_with_labels.loc[lcp_data_with_labels["labels"] == cluster].drop("labels", axis=1).reset_index(drop=True)

            # Get estimated curvature value
            curve_coef = LidarCableClustering.estimate_curvature_coefficient(subset)
            if curve_coef != -1:
                curve_coefs.append(curve_coef)

        print(f"Estimated number of cables in LiDAR cloud points: {len(curve_coefs)}")

        # for cluster in np.unique(lcp_data_with_labels["labels"][lcp_data_with_labels["labels"] != -1]):
        #     subset = lcp_data.loc[lcp_data["label"] == cluster].drop("label", axis=1).reset_index(drop=True)

        #     # Get estimated curvature value
        #     curve_coef = catenary_3d(subset)
        #     if curve_coef != -1:
        #         curve_coefs.append(curve_coef)

        # print(f"Estimated number of cables in LiDAR cloud points: {len(curve_coefs)}")
