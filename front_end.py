import streamlit as st
import cv2
import requests
from io import BytesIO, StringIO
import numpy as np
from image_vectorizer.image_processing_functions import _load_face_detection_model, _trim_image

st.write('''
# Simple image clf
## Enter the image url
''')

url = st.text_input("Enter image url")
file_path = st.file_uploader("Upload a jpg file", type="jpg")


@st.cache
model = _load_face_detection_model("./model")

def remove_head(image, model, pad=10):
    image_resize = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(
        image=image_resize, mean=(104.0, 177.0, 123.0), swapRB=True
    )
    model.setInput(blob)
    detections = model.forward()
    conf_threshold = 0.5
    h, w = image.shape[:2]
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append((x1, y1, x2, y2))
    if bboxes:
        y_limit = bboxes[0][3] + pad
        cropped = image[y_limit:, :]
        return cropped
    else:
        return image

def resize_image(image, target_size):
    if max(image.shape[:2]) > target_size:
        scale_factor = target_size / max(image.shape)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        return resized_image
    else:
        return image


def image_saver(image_path, image):
    cv2.imwrite(image_path, image)



if url:
    current_image = download_image(url)
    model_input = preprocess_image(current_image)
    prediction = model.predict(model_input)
    result = decode_result(prediction)

if file_path:
    cv2.imread(file_path)
