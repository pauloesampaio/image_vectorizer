from metaflow import FlowSpec, step


class ProcessingPipeline(FlowSpec):
    @step
    def start(self):
        """
        Load config

        """
        import pathlib
        from image_vectorizer.utils import load_config

        self.FILE_DIR = pathlib.Path(__file__).parent.absolute()
        self.config = load_config()
        self.next(self.get_file_list)

    @step
    def get_file_list(self):
        """
        Gathering all images file_pathes

        """
        import pathlib

        self.file_list = []
        for path in pathlib.Path(
            self.config["pictures_downloader"]["pictures_path"]
        ).rglob("*.jpg"):
            self.file_list.append(path.absolute().as_posix())

        print(f"Found {len(self.file_list)} images")
        self.next(self.download_face_detection_model)

    @step
    def download_face_detection_model(self):
        """
        Check if face detection model is present. If not, downloads it.
        """
        from image_vectorizer.utils import download_face_detection_model

        model_path = self.config["face_detection"]["model_path"]
        model_url = self.config["face_detection"]["model_url"]
        download_face_detection_model(model_path, model_url)
        self.next(self.remove_faces)

    @step
    def remove_faces(self):
        """
        Remove faces from images

        """
        import os
        from image_vectorizer.image_processing_functions import remove_face_multithread

        remove_face_multithread(
            self.file_list,
            os.path.join(self.FILE_DIR, self.config["face_detection"]["model_path"]),
            self.config["face_detection"]["pad"],
        )
        self.next(self.resize_pictures)

    @step
    def resize_pictures(self):
        """
        Resize images to a standard size

        """
        from image_vectorizer.image_processing_functions import resize_multithread

        resize_multithread(self.file_list, self.config["image_processing"]["max_size"])
        self.next(self.trim_pictures)

    @step
    def trim_pictures(self):
        """
        trim images

        """
        from image_vectorizer.image_processing_functions import trim_multithread

        trim_multithread(
            self.file_list,
            self.config["image_processing"]["trim_initial_variance"],
            self.config["image_processing"]["trim_delta_variance"],
            self.config["image_processing"]["trim_min_area_pct"],
        )
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    ProcessingPipeline()
