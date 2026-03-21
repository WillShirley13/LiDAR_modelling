import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


class LidarCableClustering:
    """
    Hold logic for computing
    """

    @staticmethod
    def dbscan(pts: pd.DataFrame, min_samples: int, sample_frac: float = 0.25) -> pd.DataFrame:
        """
        Typical DBSCAN implementation with additonal validation that members of clusters cannot sit ~perpendicular
        to one another in relation to the direction of the first principle component (PC1) of the dataset. That is,
        if the direction from point A -> B does sit ~perpendicular to PC1's direction, the points are likely
        members of different cables.
        Args:
            pts (pd.DataFrame): LiDAR cloud point data with columns x, y, z.
            min_samples (int): Minimum number of points within eps distance from current point for current point
            to be a core point.
            sample_frac (float): Fraxtion of dataset to be use in sample. As sample_frac -> 1, time complexity -> O(n^2).
        Returns:
            pd.DataFrame: The original pts DataFrame with an additional 'labels' column.
        """
        coords = pts[["x", "y", "z"]].to_numpy()
        n = len(coords)
        labels = np.full(n, -1)  # -1 = unvisited/noise
        cluster_id = 0
        pc1 = LidarCableClustering._get_principal_component(coords)

        # Maximum distance between two points for them to be considered in the same cluseter
        eps = LidarCableClustering._max_distance_to_nearest_neighbor(lcp_data=pts, sample_frac=sample_frac)

        def region_query(idx):
            """
            Finds all points within neighbourhood of current point
            Args:
                idx (_type_): idx of the current point being inspected for neighbours

            Returns:
                np.ndarray: Array of all points within eps distance from current point
            """
            euc_norm_dists = np.linalg.norm(coords - coords[idx], axis=1)
            return np.where(euc_norm_dists <= eps)[0]

        def are_on_same_cable(current_point_idx: int, neighbour_idxs: np.ndarray, pc1: np.ndarray) -> np.ndarray:
            """
            Eliminates all neigbours suspected of being on a different cable.
            The first principal component all points in the direction along the wires.
            If vector running from current point to neighbour is ~perpendicular to PC1
            direction, then neighbour likely part of different cable.
            Args:
                current_point_idx (int): Index of the current point being inspected for neighbours.
                neighbour_idxs (np.ndarray): Array containing indices of potential neighbours.
                pc1 (np.ndarray): First principal component of LiDAR cloud points.
            Returns:
                np.ndarray: Array containing only those neighbours on the same cable as current point.
            """

            on_same_cable = []
            for idx in neighbour_idxs:
                # Verify if the vector pointing from pts[current_point_idx] (current point)
                # to pts[idx] (neighbour) is ~perpendicular to PC1. If so, neighbour
                # is not on the same cable as pts[current_point_idx].

                # Skip if neighbour is the current point. Can't be neighbour to itself.
                if idx == current_point_idx:
                    continue

                # Vector describing direction from current point to neighbour.
                direction_to_neighbor = coords[idx] - coords[current_point_idx]
                # Alignment logic:
                # The dot product between direction_to_neighbor and pc1 measures how aligned
                # the two vectors are (whether they point in the same or opposite direction).
                # Dividing by the l2 norm of direction_to_neighbor normalises this
                # value so it reflects direction rather than magnitude.
                alignment = np.dot(direction_to_neighbor, pc1) / np.linalg.norm(direction_to_neighbor)
                # if neighbour not aligned with direction of pc1, exclude from neighbours.
                if np.abs(alignment) > 0.9:
                    on_same_cable.append(idx)
            return np.array(on_same_cable)

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

        return pts.assign(labels=labels)

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

    @staticmethod
    def _max_distance_to_nearest_neighbor(lcp_data: pd.DataFrame, sample_frac: float = 0.25) -> float:
        """
        Calculates the average distance to the nearest neighbor, using a random subset of data.
        Avoids O(n^2) search time and provides general estimate of distance between neigbours.
        To act as eps value for dbscan. Important: Assumes that in general nearest neighbour will be member
        of same cable.
        Args:
            lcp_data (pd.DataFrame): The full dataset of data to calculate the average distance to the nearest neighbor of.
            sample_frac (float): Fraction of dataset to be use in sample. As sample_frac -> 1, time complexity -> O(n^2).
        Returns:
            float - A scaled version of max distance to nearest neighbour found.
        """
        lcp_sample = lcp_data.sample(frac=sample_frac, random_state=50)

        nearest_neighbor_distances = []
        for idx, p in lcp_sample.iterrows():
            # Vectorised nearest neighbour check
            # Remove current point from consideration w/ drop()
            nearest_neighbor_distance = np.min(np.linalg.norm(lcp_data.drop(idx).values - np.array([p.x, p.y, p.z]), axis=1))
            nearest_neighbor_distances.append(nearest_neighbor_distance)

        # NOTE: In given time, could not find best way to dynamically scale neighbour distance.
        # Current value appropriate for given datasets, may not generalise well.
        return np.max(nearest_neighbor_distances) * 2

    @staticmethod
    def estimate_curvature_coefficient(cluster: pd.DataFrame, verbose: bool = False) -> float:
        """
        Given a cluster of LiDAR cloud points, the function calculates an estimate for the
        curvature of the cable represented by the cluster points.
        If the curvature value is <100 (too much slack) or >10,000 (no slack, essentially stright line), the cluster is considered erroneous and not conisdered to represent a cable.
        Args:
            cluster (pd.DataFrame): Set of cloud points for the given cluster
            verbose (bool, optional): Is set to True, will print info on calculation. Defaults to False.

        Returns:
            bool: The curvature coefficient if it meets the requirements (see above), else -1
        """
        # Only numeric columns
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

        # NOTE: The following logic re flattening the 3d points to 2d was not my own
        # In the time I had I could not solve this issue myself

        # flatten cloud points. Replace x & y coordinates with l2 norm of dist(X_i, X_start) & dist(Y_i, Y_start)
        # SQRT((X_i - X_start)^2 + (Y_i - Y_start)^2))
        lcp_flat = pd.DataFrame(
            {
                "x": cluster[["x", "y"]].apply(lambda row: np.sqrt((row["x"] - min_point["x"]) ** 2 + (row["y"] - min_point["y"]) ** 2), axis=1),
                "y": cluster["z"],
            }
        )

        # Value of x at trough
        x0 = lcp_flat.iloc[trough_idx]["x"]
        # Value of y at trough
        y0 = lcp_flat.iloc[trough_idx]["y"]

        # Wrap catenary fomrula in func
        def catenary_model(x, c):
            return y0 + c * (np.cosh((x - x0) / c) - 1)

        # Perform the fit
        popt, pcov = curve_fit(catenary_model, lcp_flat["x"], lcp_flat["y"], p0=[1000])  # p0 = initial guess for curvature 'c'

        # Curvature coefficient, c
        c_final = popt[0]

        if verbose:
            print(pd.Series(cluster_projection).describe())
            print(f"Min point: {min_point.to_dict()}")
            print(f"Max point: {max_point.to_dict()}")
            print(f"Trough point: {trough}")
            print()
            print(f"Calculated Curvature (c): {c_final:.4f}")

        # validate curvature coeficient is a reasoable value
        if 100 < c_final < 10000:
            return c_final
        else:
            return -1
