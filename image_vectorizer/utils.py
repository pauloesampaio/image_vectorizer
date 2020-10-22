import os
import requests
import pathlib
import cv2
import numpy as np
from io import BytesIO


def check_if_exists(path_to_check, create=False):
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None


def download_image(image_url):
    resp = requests.get(image_url, stream=True, timeout=5)
    im_bytes = BytesIO(resp.content)
    image = cv2.imdecode(np.fromstring(im_bytes.read(), np.uint8), -1)
    return image


def image_loader(image_location, remote=True):
    if remote:
        return download_image(image_location)
    else:
        return cv2.imread(image_location)


def resize_image(image, dest_size):
    if max(image.shape[:2]) > dest_size:
        scale_factor = dest_size / max(image.shape)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        return resized_image
    else:
        return image


def download_face_detection_model(models_path, models_url):
    check_if_exists(models_path, create=True)
    for model_url in models_url:
        model_path = os.path.join(models_path, model_url.split("/")[-1])
        if not check_if_exists(model_path):
            r = requests.get(
                model_url,
                allow_redirects=True,
            )
            with open(model_path, "wb") as f:
                f.write(r.content)
    return print("Models downloaded")


def load_face_detection_model(models_path):
    model_file = None
    model_config = None
    for path in pathlib.Path(models_path).rglob("*.pb"):
        model_file = path.absolute().as_posix()

    for path in pathlib.Path(models_path).rglob("*.pbtxt"):
        model_config = path.absolute().as_posix()

    model = cv2.dnn.readNetFromTensorflow(model_file, model_config)
    return model


def find_and_remove_faces(image, model, pad=10):
    resized_image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(
        image=resized_image, mean=(104.0, 177.0, 123.0), swapRB=True
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


def trim_image(image, var_threshold=250):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    var_x = np.var(blurred, axis=0)
    var_y = np.var(blurred, axis=1)

    limits_x = np.where(var_x > var_threshold)[0]
    if limits_x.size <= 1:
        limits_x = [0, image.shape[1]]

    limits_y = np.where(var_y > var_threshold)[0]
    if limits_y.size <= 0:
        limits_y = [0, image.shape[0]]

    lim_left, lim_right = limits_x[0], limits_x[-1]
    lim_top, lim_bottom = limits_y[0], limits_y[-1]
    return image[lim_top:lim_bottom, lim_left:lim_right]


def trim_iterative(image, initial_threshold=250):
    delta_threshold = 25
    min_area_pct = 0.25
    threshold = initial_threshold
    original_x = image.shape[1]
    original_y = image.shape[0]
    done = False
    current_iteration = None
    while not done:
        current_iteration = trim_image(image, threshold)
        if (current_iteration.shape[1] / original_x < min_area_pct) or (
            current_iteration.shape[0] / original_y < min_area_pct
        ):
            threshold = threshold - delta_threshold
        elif (current_iteration.shape[1] / original_x == 1.0) or (
            current_iteration.shape[0] / original_y == 1.0
        ):
            threshold = threshold - delta_threshold
        else:
            done = True
        if threshold <= 0:
            done = True
            current_iteration = image
    return current_iteration
