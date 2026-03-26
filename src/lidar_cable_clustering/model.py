from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


class CableResult:
    """Structured output from the cable identification pipeline."""

    def __init__(
        self,
        labeled_data: pd.DataFrame,
        cables: List[Dict],
        estimated_cables: pd.DataFrame,
    ):
        self.labeled_data = labeled_data
        self.cables = cables
        self.estimated_cables = estimated_cables

    @property
    def cable_count(self) -> int:
        return len(self.cables)

    @property
    def labels(self) -> pd.Series:
        return pd.Series(self.labeled_data["labels"])

    @property
    def curvature_coefficients(self) -> Dict[int, float]:
        return {c["cluster_id"]: c["curvature_coef"] for c in self.cables}


class LidarCableClustering:
    """
    Holds logic for computing the curvature coefficient of a cable represented by a cluster of LiDAR cloud points as well
    as estimating the number of cables present in a LiDAR cloud points dataset.
    """

    # Threshold for the alignment of a point to the principal component
    ALLIGNMENT_THRESHOLD: float
    # Minimum and maximum curvature coefficients for a cable
    MIN_CURVATURE_COEFFICIENT: int
    MAX_CURVATURE_COEFFICIENT: int
    # Scale factor for the nearest neighbour distance
    NEIGHBOUR_DISTANCE_SCALE_FACTOR: float
    # Random state for the sample
    RANDOM_STATE: int
    # Initial guess for the curvature coefficient
    INITIAL_CURVATURE_COEFFICIENT_GUESS: int
    # Number of points to sample for the nearest neighbour distance
    SAMPLE_FRACTION: float
    # Minimum number of points within eps distance from current point for current point to be a core point
    MIN_SAMPLES: int

    def __init__(
        self,
        lcp_data: pd.DataFrame,
        allignment_threshold: float = 0.8,
        min_curvature_coefficient: int = 100,
        max_curvature_coefficient: int = 10000,
        neighbour_distance_scale_factor: float = 3,
        random_state: int = 59,
        initial_curvature_coefficient_guess: int = 300,
        sample_fraction: float = 0.2,
        min_samples: int = 2,
    ):
        """
        Initialises the LidarCableClustering class.
        Args:
            - lcp_data (pd.DataFrame): The LiDAR cloud point data.
            - allignment_threshold (float): The threshold for the alignment of a point to the principal component.
            - min_curvature_coefficient (int): The minimum curvature coefficient for a cable.
            - max_curvature_coefficient (int): The maximum curvature coefficient for a cable.
            - neighbour_distance_scale_factor (int): The scale factor for the nearest neighbour distance.
            - random_state (int): The random state for the sample.
            - initial_curvature_coefficient_guess (int): The initial guess for the curvature coefficient.
            - sample_fraction (float): The fraction of the data to sample for the nearest neighbour distance.
              (Note: As sample_frac -> 1, clustering improves but time complexity -> O(n^2)).
            - min_samples (int): The minimum number of points within eps distance from current point for current point to be a core point.
        """
        self.lcp_data = lcp_data
        self.ALLIGNMENT_THRESHOLD = allignment_threshold
        self.MIN_CURVATURE_COEFFICIENT = min_curvature_coefficient
        self.MAX_CURVATURE_COEFFICIENT = max_curvature_coefficient
        self.NEIGHBOUR_DISTANCE_SCALE_FACTOR = neighbour_distance_scale_factor
        self.RANDOM_STATE = random_state
        self.INITIAL_CURVATURE_COEFFICIENT_GUESS = initial_curvature_coefficient_guess
        self.SAMPLE_FRACTION = sample_fraction
        self.MIN_SAMPLES = min_samples

    def identify_cables(self, verbose: bool = False) -> CableResult:
        """
        Full pipeline: clusters the point cloud via (adapted) DBSCAN, then filters
        clusters by curvature coefficient to identify actual cables.
        Args:
            verbose (bool): Forward to estimate_curvature_coefficient.
        Returns:
            CableResult containing labeled data, per-cable info, and estimated cable points.
        """
        # Step 1: Cluster the point cloud using DBSCAN with alignment filtering
        labeled_data = self._dbscan()

        cables: List[Dict] = []
        all_estimated_cables = pd.DataFrame()

        # Step 2: Iterate over each non-noise cluster and validate via curvature
        for cluster_id in np.unique(labeled_data["labels"][labeled_data["labels"] != -1]):
            # Isolate points belonging to this cluster
            subset = labeled_data.loc[labeled_data["labels"] == cluster_id].drop("labels", axis=1).reset_index(drop=True)

            # Fit catenary curve and retrieve curvature coefficient
            curve_coef, estimated_cable = self._estimate_curvature_coefficient(subset, verbose=verbose)

            # Keep cluster only if curvature falls within plausible cable range
            if curve_coef != -1 and estimated_cable is not None:
                cables.append({"cluster_id": cluster_id, "curvature_coef": curve_coef})
                estimated_cable["label"] = cluster_id
                all_estimated_cables = pd.concat([all_estimated_cables, estimated_cable], ignore_index=True)

        return CableResult(
            labeled_data=labeled_data,
            cables=cables,
            estimated_cables=all_estimated_cables,
        )

    # NOTE: I did not write the base DBSCAN model code. I took the typical implementation and added
    # the additional validation that members of clusters cannot sit ~perpendicular to one another in
    # relation to the direction of the PC1 of the dataset.
    def _dbscan(self) -> pd.DataFrame:
        """
        Typical DBSCAN implementation with additonal validation that members of clusters cannot sit ~perpendicular
        to one another in relation to the direction of the first principle component (PC1) of the dataset. That is,
        if the direction from point A -> B does sit ~perpendicular to PC1's direction, the points are likely
        members of different cables.
        Returns:
            pd.DataFrame: The original lcp_data DataFrame with an additional 'labels' column.
        """
        coords = self.lcp_data[["x", "y", "z"]].to_numpy()
        n = len(coords)
        labels = np.full(n, -1)  # -1 = unvisited/noise
        cluster_id = 0
        alignment_threshold = self.ALLIGNMENT_THRESHOLD
        min_samples = self.MIN_SAMPLES

        # Get principal component of dataset
        pc1 = LidarCableClustering._get_principal_component(coords)

        # Maximum distance between two points for them to be considered in the same cluster
        eps = self._max_distance_to_nearest_neighbour()

        def region_query(idx):
            """
            Finds all points within neighbourhood of current point
            Args:
                idx (int): idx of the current point being inspected for neighbours

            Returns:
                np.ndarray: Array of all the indexes of points within eps distance from current point
            """
            euc_norm_dists = np.linalg.norm(coords - coords[idx], axis=1)
            # return just indexes of neighbours
            return np.where(euc_norm_dists <= eps)[0]

        def are_on_same_cable(current_point_idx: int, neighbour_idxs: np.ndarray, pc1: np.ndarray) -> np.ndarray:
            """
            Eliminates all neigbours suspected of being on a different cable.
            The first principal component always points (on the x & y dimensions) in the direction along the wires.
            If vector running from current point to neighbour is ~perpendicular to PC1
            direction, then neighbour likely part of different cable.
            Args:
                current_point_idx (int): Index of the current point being inspected for neighbours.
                neighbour_idxs (np.ndarray): Array containing indices of potential neighbours.
                pc1 (np.ndarray): First principal component of LiDAR cloud points.
            Returns:
                np.ndarray: Array containing the indexes of only those neighbours on the same cable as current point.
            """

            # Get direction from current point to each potential neighbour
            direction_to_neighbours = coords[:, 0:2][neighbour_idxs] - coords[:, 0:2][current_point_idx]

            norms = np.linalg.norm(direction_to_neighbours, axis=1)
            # Mask for removing the current point from its own neighbours
            not_self = norms > 0
            valid_neighbour_idxs = neighbour_idxs[not_self]

            # Get normalised alignment of direction to neighbours and pc1
            alignment = np.dot(direction_to_neighbours[not_self], pc1[0:2]) / (norms[not_self] * np.linalg.norm(pc1[0:2]))

            # If alignment is greater than threshold, the neighbour is on the same cable as the current point
            on_same_cable = np.abs(alignment) > alignment_threshold
            return valid_neighbour_idxs[on_same_cable]

        for i in range(n):
            if labels[i] != -1:
                continue

            potential_neighbours = region_query(i)

            # Filter for neighbours geometrically aligned with cable of current point
            neighbours = are_on_same_cable(i, potential_neighbours, pc1)

            if len(neighbours) < min_samples:
                labels[i] = -1  # noise
                continue

            labels[i] = cluster_id
            seed_set = set(neighbours) - {i}

            while seed_set:
                j = seed_set.pop()
                if labels[j] != -1:
                    continue
                labels[j] = cluster_id
                potential_new_neighbours = region_query(j)
                new_neighbours = are_on_same_cable(j, potential_new_neighbours, pc1)
                if len(new_neighbours) >= min_samples:
                    seed_set.update(new_neighbours)

            cluster_id += 1

        # Return original data with cluster labels column
        return self.lcp_data.assign(labels=labels)

    @staticmethod
    def _get_principal_component(lcp_data: np.ndarray) -> np.ndarray:
        """
        Gets the first principal component of the passed data
        Args:
            lcp_data: pd.DataFrame - The data to get the principal component of
        Returns:
            np.array - The first principal component of the cloud points data
        """
        pca = PCA(n_components=3)
        pca.fit(lcp_data)
        return np.array(pca.components_[0])

    def _max_distance_to_nearest_neighbour(self) -> float:
        """
        Calculates the average distance to the nearest neighbor, using a random subset of data.
        Avoids O(n^2) search time and provides general estimate of distance between neighbours.
        To act as eps value for dbscan. Important: Assumes that in general nearest neighbour will be member
        of same cable.
        Returns:
            float - A scaled version of max distance to nearest neighbour found.
        """
        lcp_sample = self.lcp_data.sample(frac=self.SAMPLE_FRACTION)  # random_state=self.RANDOM_STATE)

        nearest_neighbor_distances = []
        for idx, p in lcp_sample.iterrows():
            # Vectorised nearest neighbour check
            # Remove current point from consideration w/ drop()
            nearest_neighbor_distance = np.min(np.linalg.norm(self.lcp_data.drop(idx).values - np.array([p.x, p.y, p.z]), axis=1))
            nearest_neighbor_distances.append(nearest_neighbor_distance)

        # NOTE: In given time, could not find best way to dynamically scale neighbour distance.
        # Current value appropriate for given datasets, may not generalise well.
        return np.max(nearest_neighbor_distances) * self.NEIGHBOUR_DISTANCE_SCALE_FACTOR

    def _estimate_curvature_coefficient(self, cluster: pd.DataFrame, verbose: bool = False) -> Tuple[float, pd.DataFrame | None]:
        """
        Given a cluster of LiDAR cloud points, the function calculates an estimate for the
        curvature of the cable represented by the cluster points.
        If the curvature value is outside the configured bounds, the cluster is considered
        erroneous and does not represent a cable.
        Args:
            cluster (pd.DataFrame): Set of cloud points for the given cluster
            verbose (bool, optional): Is set to True, will print info on calculation. Defaults to False.

        Returns:
            Tuple[float, pd.DataFrame]:
                - index 0: The curvature coefficient if it meets the requirements (see above), else -1
                - index 1: DataFrame containing the points estimating the cable curve, else None if coef invalid
        """
        # Remove labels column, if present
        coords = cluster[["x", "y", "z"]].to_numpy()

        # Find trough (lowest z) and it's index
        trough = cluster.loc[cluster["z"] == cluster["z"].min()]
        trough_idx = cluster["z"].idxmin()

        # Get principal component
        cluster_pc1 = LidarCableClustering._get_principal_component(coords)

        # Project points onto PC1. Mean centre data first.
        mean = coords.mean(axis=0)
        cluster_projection = (coords - mean) @ cluster_pc1

        # Find extreme points of projected points (indicate ends of cable)
        max_idx = cluster_projection.argmax()
        min_idx = cluster_projection.argmin()
        min_point = cluster.iloc[min_idx]
        max_point = cluster.iloc[max_idx]

        # flatten cloud points. Replace x & y coordinates with l2 norm of dist(X_i, X_start) & dist(Y_i, Y_start)
        # SQRT((X_i - X_start)^2 + (Y_i - Y_start)^2))
        lcp_flat = pd.DataFrame(
            {
                "x": cluster[["x", "y"]].apply(
                    lambda row: np.sqrt((row["x"] - min_point["x"]) ** 2 + (row["y"] - min_point["y"]) ** 2),
                    axis=1,
                ),
                "y": cluster["z"],
            }
        )

        # Value of x at trough
        x0 = lcp_flat.iloc[trough_idx]["x"]
        # Value of y at trough
        y0 = lcp_flat.iloc[trough_idx]["y"]

        # Wrap catenary formula in func
        def catenary_model(x, c):
            return y0 + c * (np.cosh((x - x0) / c) - 1)

        # Perform the fit
        popt, _ = curve_fit(
            catenary_model,
            lcp_flat["x"],
            lcp_flat["y"],
            p0=[self.INITIAL_CURVATURE_COEFFICIENT_GUESS],
        )

        # Curvature coefficient, c
        c_final = popt[0]

        if verbose:
            print(pd.Series(cluster_projection).describe())
            print(f"Min point: {min_point.to_dict()}")
            print(f"Max point: {max_point.to_dict()}")
            print(f"Trough point: {trough}")
            print()
            print(f"Calculated Curvature (c): {c_final:.4f}")

        # validate curvature coeficient is a reasonable value
        if self.MIN_CURVATURE_COEFFICIENT < c_final < self.MAX_CURVATURE_COEFFICIENT:
            # get flattened x,y values to get estimated z value via catenary_model()
            dist_from_start = lcp_flat["x"]
            estimated_z_values = catenary_model(dist_from_start, c_final)

            # Sort values for plotting line later
            # Line plotting none sequential data points causes zig-zag line
            sort_idx = dist_from_start.argsort()
            # Df containing og x & y values, with estimated z values.
            estimated_cable = pd.DataFrame(
                {
                    "x": cluster["x"].iloc[sort_idx].values,
                    "y": cluster["y"].iloc[sort_idx].values,
                    "z": estimated_z_values.iloc[sort_idx].values,
                }
            )
            return (c_final, estimated_cable)
        else:
            return (-1, None)
