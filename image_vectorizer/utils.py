import os
import pathlib
import numpy as np
import yaml
import glob
import pandas as pd


def load_config(config_path="./config/config.yml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def check_if_exists(path_to_check, create=False):
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None
    else:
        return True


def get_paths_dataframe(pictures_path, infer_classes=False):
    path_list = glob.glob(
        os.path.join(pictures_path, "**", "*.jpg"),
        recursive=True,
    )
    paths_dataframe = pd.DataFrame(path_list, columns=["filename"])
    if infer_classes:
        paths_dataframe["class"] = [
            w.split("/")[-2] for w in paths_dataframe["filename"]
        ]
    return paths_dataframe


def save_array(filepath, array):
    check_if_exists(os.path.dirname(filepath), create=True)
    with open(filepath, "wb") as f:
        np.save(f, array)
    return None
