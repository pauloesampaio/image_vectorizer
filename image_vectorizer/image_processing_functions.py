import cv2
from .utils import (
    image_loader,
    load_face_detection_model,
    resize_image,
    find_and_remove_faces,
    trim_iterative,
)
from concurrent.futures import ThreadPoolExecutor

MODEL = load_face_detection_model("../model")


def _resize_multithread_helper(image_combo):
    file_path, dest_size = image_combo
    image = image_loader(file_path, remote=False)
    return cv2.imwrite(file_path, resize_image(image, dest_size))


def resize_multithread(image_list, dest_size):
    image_list = [(w,) + (dest_size,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(_resize_multithread_helper, image_list)
    return None


def _remove_face_multithread_helper(image_combo):
    file_path, model_path, pad = image_combo
    model = load_face_detection_model(model_path)
    image = image_loader(file_path, remote=False)
    return cv2.imwrite(file_path, find_and_remove_faces(image, model, pad))


def remove_face_multithread(image_list, face_detection_model_path, pad):
    image_combo_list = [
        (w,)
        + (
            face_detection_model_path,
            pad,
        )
        for w in image_list
    ]
    with ThreadPoolExecutor() as executor:
        executor.map(_remove_face_multithread_helper, image_combo_list)
    return None


def _trim_multithread_helper(image_combo):
    file_path, initial_variance = image_combo
    image = image_loader(file_path, remote=False)
    return cv2.imwrite(file_path, trim_iterative(image, initial_variance))


def trim_multithread(image_list, var_threshold):
    image_list = [(w,) + (var_threshold,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(_trim_multithread_helper, image_list)
    return None
