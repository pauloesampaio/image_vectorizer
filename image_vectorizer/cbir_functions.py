import nmslib
import os
import cv2


def get_file_list(file_list_path):
    with open(file_list_path) as f:
        file_list = f.read().splitlines()
    return file_list


def build_ann_model(vectors, init_config, index_time_config, model_path):
    ann_index = nmslib.init(**init_config)
    ann_index.addDataPointBatch(vectors)
    ann_index.createIndex(index_time_config)
    ann_index.saveIndex(model_path)
    return print(f"Saved model to {model_path}")


def load_ann_model(model_path, init_config, query_time_config):
    ann_index = nmslib.init(**init_config)
    ann_index.loadIndex(model_path)
    ann_index.setQueryTimeParams(query_time_config)
    return ann_index


def query_index(ann_index, query_vector, k=10):
    k_index, k_dist = ann_index.knnQuery(query_vector, k=k)
    return {"neighbours": k_index, "distances": k_dist}


def get_results(file_list, ann_result):
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
