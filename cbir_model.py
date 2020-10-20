import numpy as np
import nmslib
import os
import cv2
from image_vectorizer.images_download_functions import _download_image


nmslib_config = {
    "init": {
        "method": "hnsw",
        "space": "cosinesimil",
        "data_type": nmslib.DataType.DENSE_VECTOR,
    },
    "index_time": {"M": 15, "indexThreadQty": 4, "efConstruction": 100},
    "query_time": {
        "efSearch": 100,
    },
}

vectors = np.load("./vectors.npy")
with open("file_list.txt") as f:
    file_list = f.read().splitlines()


def _create_index(vectors_array, nmslib_config):
    ann_index = nmslib.init(**nmslib_config["init"])
    ann_index.addDataPointBatch(vectors)
    ann_index.createIndex(nmslib_config["index_time"])
    ann_index.setQueryTimeParams(nmslib_config["query_time"])
    return ann_index


def _query_index(ann_index, query_vector, k=10):
    k_index, k_dist = ann_index.knnQuery(query_vector, k=k)
    return {"neighbours": k_index, "distances": k_dist}


def _get_results(file_list, ann_result):
    results = []
    for k_index, k_distance in zip(ann_result["neighbours"], ann_result["distances"]):
        current_result = {}
        file_name = file_list[k_index]
        current_result["product"], current_result["retailer"] = file_name.split("/")[
            ::-1
        ][:2]
        current_result["image"] = cv2.cvtColor(
            cv2.imread(os.path.join("./pictures", file_name)), cv2.COLOR_BGR2RGB
        )
        current_result["distance"] = k_distance
        results.append(current_result)
    return results
