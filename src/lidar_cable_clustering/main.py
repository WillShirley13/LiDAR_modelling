import pandas as pd

from lidar_cable_clustering.model import LidarCableClustering
from lidar_cable_clustering.utils import (
    cluster_stats,
    load_data,
    plot_clusters,
    plot_estimated_cable,
)


def main():
    difficulty = None
    while True:
        difficulty = (
            input(
                "\nPlease provide a dataset difficulty to model or 'exit' to quit. \nOptions: \n- easy\n- medium\n- hard\n- extrahard\n(You may append _180, _90cw, _90acw to the difficulty to apply data augmentation. e.g. easy_180)\nInput: "
            )
            .strip()
            .lower()
        )
        if difficulty == "exit":
            break
        # Verify difficulty input
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

        sample_frac = input(
            """\nPlease provide value between 0.01 & 1.0 (Default = 0.2). This is the size (%) of the sample used to estimate distance between neighbours.
            (Note: As sample_frac -> 1, clustering improves but time complexity -> O(n^2))\nInput:"""
        )

        # Verify sample input
        try:
            sample_frac = float(sample_frac)
            if 0 > sample_frac > 1:
                raise ValueError("Invalid sample size input. Must be between 0.01 and 1.0. Enter again")
        except ValueError:
            print("Invalid sample size input. Must be between 0.01 and 1.0. Enter again")
            continue

        # Load in requested data
        lcp_data: pd.DataFrame = load_data(difficulty)

        # Run full pipeline: clustering + curvature filtering
        model = LidarCableClustering(lcp_data, sample_fraction=sample_frac)
        result = model.identify_cables()

        cluster_stats(result.labels)
        plot_clusters(result.labeled_data, result.labels, difficulty, sample_frac)

        for cable in result.cables:
            print(f"\nCluster {cable['cluster_id'] + 1} has curvature coefficient: {cable['curvature_coef']}")

        print(f"\nEstimated number of cables in LiDAR cloud points: {result.cable_count}")

        plot_estimated_cable(result.estimated_cables, difficulty, sample_frac)

    print("Thank you for using my Lidar Cable Clustering Model!")


if __name__ == "__main__":
    main()
