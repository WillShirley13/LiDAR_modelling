# Augment provided data by rotating to provide synthetic LiDAR cable points datasets

from pathlib import Path

import pandas as pd

from lidar_cable_clustering.utils import load_data

# NOTE: None of the below augmentation methods stress the model as much as more rigorous augmentation methods would.
# as they preserve overall structure of the data. However, they are a good starting point for time I had.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def rotate_180_lcp_data(lcp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Flip/Reflect the data across the z axis
    Args:
        lcp_data (pd.DataFrame): The full LiDAR cloud point dataset

    Returns:
        pd.DataFrame: The augmented dataset
    """
    return pd.DataFrame({"x": -lcp_data["x"], "y": -lcp_data["y"], "z": lcp_data["z"]}, index=lcp_data.index)


def rotate_90_cw_lcp_data(lcp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rotate the data 90 degrees clockwise across the z axis
    Args:
        lcp_data (pd.DataFrame): The full LiDAR cloud point dataset

    Returns:
        pd.DataFrame: The augmented dataset
    """
    return pd.DataFrame({"x": lcp_data["y"], "y": -lcp_data["x"], "z": lcp_data["z"]}, index=lcp_data.index)


def rotate_90_acw_lcp_data(lcp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rotate the data 90 degrees anti-clockwise across the z axis
    Args:
        lcp_data (pd.DataFrame): The full LiDAR cloud point dataset

    Returns:
        pd.DataFrame: The augmented dataset
    """
    return pd.DataFrame({"x": -lcp_data["y"], "y": lcp_data["x"], "z": lcp_data["z"]}, index=lcp_data.index)


if __name__ == "__main__":
    for dif in ["easy", "medium", "hard", "extrahard"]:
        data = load_data(dif)
        data_180 = rotate_180_lcp_data(data)
        data_90_cw = rotate_90_cw_lcp_data(data)
        data_90_acw = rotate_90_acw_lcp_data(data)

        data_180.to_parquet(DATA_DIR / f"synthetic_data/lidar_cable_points_{dif}_180.parquet")
        data_90_cw.to_parquet(DATA_DIR / f"synthetic_data/lidar_cable_points_{dif}_90cw.parquet")
        data_90_acw.to_parquet(DATA_DIR / f"synthetic_data/lidar_cable_points_{dif}_90acw.parquet")
