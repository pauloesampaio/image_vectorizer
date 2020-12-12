import os
import pathlib
import numpy as np
import yaml
import glob
import pandas as pd


def load_config(config_path="./config/config.yml"):
    """Simple helper to load configuration file

    Args:
        config_path (str, optional): Path to config.yml. Defaults to "./config/config.yml".

    Returns:
        [dict]: Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def check_if_exists(path_to_check, create=False):
    """Check if a path exists. Optionally, creates if needed.

    Args:
        path_to_check (string): Path to be checked
        create (bool, optional): Create or not if needed. Defaults to False.

    Returns:
        None
    """
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None
    else:
        return True


def get_paths_dataframe(pictures_path, infer_classes=False):
    """Get path of images (jpg, jpeg and png). In infer_classes, uses parent folder
    as class name.

    Args:
        pictures_path (string): path to images folder
        infer_classes (bool, optional): If true, use parent folder as class. Defaults to False.

    Returns:
        [pd.DataFrame]: Dataframe with file paths (and classes)
    """
    file_types = ["*.jpg", "*.jpeg", "*.png"]
    path_list = []
    for extension in file_types:
        current_paths = glob.glob(
            os.path.join(pictures_path, "**", extension),
            recursive=True,
        )
        path_list.extend(current_paths)
    paths_dataframe = pd.DataFrame(path_list, columns=["filename"])
    if infer_classes:
        paths_dataframe["class"] = [
            w.split("/")[-2] for w in paths_dataframe["filename"]
        ]
    return paths_dataframe


def save_array(filepath, array):
    """Save numpy array to file

    Args:
        filepath (str): Path to save array
        array (np.array): Array to be saved

    Returns:
        None
    """
    check_if_exists(os.path.dirname(filepath), create=True)
    with open(filepath, "wb") as f:
        np.save(f, array)
    return None
