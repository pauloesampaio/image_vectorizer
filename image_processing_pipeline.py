import os
import pathlib
from metaflow import FlowSpec, step, Parameter


class ProcessingPipeline(FlowSpec):
    FILE_DIR = pathlib.Path(__file__).parent.absolute()

    PICTURES_PATH = Parameter(
        "pictures_path", default=os.path.join(FILE_DIR, "pictures")
    )

    MODELS_PATH = Parameter("models_path", default=os.path.join(FILE_DIR, "model"))

    MODELS_URL = Parameter(
        "models_url",
        default=[
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb",  # noqa
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",  # noqa
        ],
    )

    FACE_PAD = Parameter("face_pad", default=10)
    IMAGE_SIZE = Parameter("image_size", default=448)
    TRIM_THRESHOLD = Parameter("trim_threshol", default=250)

    @step
    def start(self):
        """
        Gathering all images file_pathes

        """
        from pathlib import Path

        self.file_list = []
        for path in Path(self.PICTURES_PATH).rglob("*.jpg"):
            self.file_list.append(path.absolute().as_posix())

        print(f"Found {len(self.file_list)} images")
        self.next(self.download_face_detection_model)

    @step
    def download_face_detection_model(self):
        """
        Download face detection model

        """
        from image_vectorizer.utils import (
            download_face_detection_model,
        )

        download_face_detection_model(self.MODELS_PATH, self.MODELS_URL)
        self.next(self.remove_faces)

    @step
    def remove_faces(self):
        """
        Remove faces from images

        """
        from image_vectorizer.image_processing_functions import remove_face_list

        remove_face_list(self.file_list, self.MODELS_PATH, self.FACE_PAD)
        self.next(self.resize_pictures)

    @step
    def resize_pictures(self):
        """
        Resize images to a standard size

        """
        from image_vectorizer.image_processing_functions import resize_image_list

        resize_image_list(self.file_list, self.IMAGE_SIZE)
        self.next(self.trim_pictures)

    @step
    def trim_pictures(self):
        """
        trim images

        """
        from image_vectorizer.image_processing_functions import trim_image_list

        trim_image_list(self.file_list, self.TRIM_THRESHOLD)
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    ProcessingPipeline()
