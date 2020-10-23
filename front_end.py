import streamlit as st
import cv2
from image_vectorizer.image_vectorizing_functions import load_vectorizer_model
from image_vectorizer.utils import (
    load_face_detection_model,
    image_loader,
    resize_image,
    trim_iterative,
    find_and_remove_faces,
)

from image_vectorizer.cbir_model import (
    load_ann_model,
    get_file_list,
    query_index,
    get_results,
)

model_path = "./model"
ann_model_path = "./model/ann_index.hsnw"
file_list_path = "./file_list.txt"
pad = 10
dest_size = 448
initial_threshold = 250


st.write(
    """
# Simple image clf
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
def cached_load_ann_model(ann_model_path):
    ann_model = load_ann_model(ann_model_path)
    return ann_model


face_detector_model = load_face_detection_model(model_path)
vectorizer_model = cached_load_vectorizer()
ann_model = cached_load_ann_model(ann_model_path)
file_list = cached_load_file_list(file_list_path)

col1, col2 = st.beta_columns(2)

if url:
    current_image = image_loader(url, remote=True)
    col1.header("Original")
    col1.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), width=128)
    no_face_image = find_and_remove_faces(current_image, face_detector_model, pad)
    resized_image = resize_image(no_face_image, dest_size)
    trimmed_image = trim_iterative(resized_image, initial_threshold)
    rgb_image = cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2RGB)
    col1.header("Processed")
    col1.image(rgb_image, width=128)
    vectorizer_input = cv2.resize(rgb_image, (224, 224)).reshape((1, 224, 224, 3))
    feature_vector = vectorizer_model.predict(vectorizer_input)
    results = get_results(file_list, query_index(ann_model, feature_vector))
    col2.header("Matches")
    for each in results:
        col2.image(each["image"], width=128)
        col2.write(each["product"])
        col2.write(each["distance"])
