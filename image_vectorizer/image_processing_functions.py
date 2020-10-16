import os
import cv2
import requests
import pathlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _check_if_exists(path_to_check, create=False):
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None


def _resize_image(image_combo):
    image_path, dest_size = image_combo
    image = cv2.imread(image_path)
    if max(image.shape[:2]) > dest_size:
        scale_factor = dest_size / max(image.shape)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        cv2.imwrite(image_path, resized_image)
    return None


def resize_images(image_list, dest_size):
    image_list = [(w,) + (dest_size,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(_resize_image, image_list)
    return None


def download_face_detection_model(models_path, models_url):
    _check_if_exists(models_path, create=True)
    for model_url in models_url:
        model_path = os.path.join(models_path, model_url.split("/")[-1])
        if not _check_if_exists(model_path):
            r = requests.get(
                model_url,
                allow_redirects=True,
            )
            with open(model_path, "wb") as f:
                f.write(r.content)
    return print("Models downloaded")


def _load_face_detection_model(models_path):
    model_file, model_config = None
    for path in pathlib.Path(models_path).rglob("*.pb"):
        model_file = path.absolute().as_posix()

    for path in pathlib.Path(models_path).rglob("*.pbtxt"):
        model_config = path.absolute().as_posix()

    model = cv2.dnn.readNetFromTensorflow(model_file, model_config)
    return model


def _find_and_remove_faces(image_combo):
    image_path, models_path, pad = image_combo
    model = _load_face_detection_model(models_path)
    image = cv2.imread(image_path)
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
        cv2.imwrite(image_path, cropped)
    return None


def face_remover(image_list, face_detection_model_path, pad):
    image_combo_list = [
        (w,)
        + (
            face_detection_model_path,
            pad,
        )
        for w in image_list
    ]
    with ThreadPoolExecutor() as executor:
        executor.map(_find_and_remove_faces, image_combo_list)
    return None


def _trim_image(image, var_threshold=250):
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


def _trime_iterative(image_combo):
    image_path, initial_threshold = image_combo
    image = cv2.imread(image_path)
    delta_threshold = 25
    min_area_pct = 0.25
    max_area_pct = 0.99
    threshold = initial_threshold
    original_x = image.shape[1]
    original_y = image.shape[0]
    done = False
    current_iteration = None
    while not done:
        current_iteration = _trim_image(image, threshold)
        if (current_iteration.shape[1] / original_x < min_area_pct) or (
            current_iteration.shape[0] / original_y < min_area_pct
        ):
            threshold = threshold - delta_threshold
        if (current_iteration.shape[1] / original_x >= max_area_pct) or (
            current_iteration.shape[0] / original_y >= max_area_pct
        ):
            threshold = threshold - delta_threshold
        elif threshold < 0:
            done = True
            current_iteration = image
        else:
            done = True
    cv2.imwrite(image_path, current_iteration)
    return None


def trim_images(image_list, var_threshold):
    image_list = [(w,) + (var_threshold,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(_trime_iterative, image_list)
    return None
