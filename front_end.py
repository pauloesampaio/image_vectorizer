import streamlit as st
import cv2
from image_vectorizer.image_vectorizing_functions import load_vectorizer_model
from image_vectorizer.utils import (
    load_config,
    load_face_detection_model,
    image_loader,
    resize_image,
    trim_iterative,
    find_and_remove_faces,
)

from image_vectorizer.cbir_functions import (
    load_ann_model,
    get_file_list,
    query_index,
    get_results,
)
import matplotlib.pyplot as plt

config = load_config()


st.write(
    """
# Vanilla image matcher
## Enter the image url
"""
)

url = st.text_input("Enter image url")


@st.cache
def cached_load_vectorizer():
    vectorizer_model = load_vectorizer_model()
    return vectorizer_model


@st.cache
def cached_load_file_list(file_list_path):
    file_list = get_file_list(file_list_path)
    return file_list


@st.cache
def cached_load_ann_model(
    ann_model_path, ann_init_configuration, ann_query_time_configuration
):
    ann_model = load_ann_model(
        ann_model_path, ann_init_configuration, ann_query_time_configuration
    )
    return ann_model


def make_result_grid_matplotlib(results):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    i = 0
    for r in range(0, 3):
        for c in range(0, 3):
            product_id = results[i]["product"].split("_")[0]
            distance = str(round(results[i]["distance"], 4))
            axs[r, c].imshow(results[i]["image"])
            axs[r, c].set_axis_off()
            axs[r, c].title.set_text(f"id: {product_id} \n distance: {distance}")
            i = i + 1
    fig.tight_layout()
    return fig


face_detector_model = load_face_detection_model(config["face_detection"]["model_path"])
vectorizer_model = cached_load_vectorizer()
ann_model = cached_load_ann_model(
    config["ann_model"]["ann_model_path"],
    config["ann_model"]["init"],
    config["ann_model"]["query_time"],
)
file_list = cached_load_file_list(config["image_vectorizer"]["file_list_path"])

col1, col2 = st.beta_columns(2)

if url:
    current_image = image_loader(url, remote=True)
    col1.header("Original")
    col1.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), width=128)
    no_face_image = find_and_remove_faces(
        current_image, face_detector_model, config["face_detection"]["pad"]
    )
    resized_image = resize_image(no_face_image, config["image_processing"]["max_size"])
    trimmed_image = trim_iterative(
        resized_image,
        config["image_processing"]["trim_initial_variance"],
        config["image_processing"]["trim_delta_variance"],
        config["image_processing"]["trim_min_area_pct"],
    )
    rgb_image = cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2RGB)
    col1.header("Processed")
    col1.image(rgb_image, width=128)
    vectorizer_input = cv2.resize(rgb_image, (224, 224)).reshape((1, 224, 224, 3))
    feature_vector = vectorizer_model.predict(vectorizer_input)
    results = get_results(file_list, query_index(ann_model, feature_vector))
    col2.header("Matches")
    col2.pyplot(make_result_grid_matplotlib(results))
