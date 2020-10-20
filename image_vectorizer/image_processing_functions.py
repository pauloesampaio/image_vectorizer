from .utils import resize_image, find_and_remove_faces, trim_iterative
from concurrent.futures import ThreadPoolExecutor


def resize_image_list(image_list, dest_size):
    image_list = [(w,) + (dest_size,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(resize_image, image_list)
    return None


def remove_face_list(image_list, face_detection_model_path, pad):
    image_combo_list = [
        (w,)
        + (
            face_detection_model_path,
            pad,
        )
        for w in image_list
    ]
    with ThreadPoolExecutor() as executor:
        executor.map(find_and_remove_faces, image_combo_list)
    return None


def trim_image_list(image_list, var_threshold):
    image_list = [(w,) + (var_threshold,) for w in image_list]
    with ThreadPoolExecutor() as executor:
        executor.map(trim_iterative, image_list)
    return None
