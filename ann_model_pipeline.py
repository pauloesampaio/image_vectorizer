import os
from metaflow import FlowSpec, step


class DownloadingPipeline(FlowSpec):
    @step
    def start(self):
        """
        Load config
        """
        import pathlib
        from image_vectorizer.utils import load_config

        self.FILE_DIR = pathlib.Path(__file__).parent.absolute()
        self.config = load_config()
        self.next(self.load_vectors)

    @step
    def load_vectors(self):
        """
        Load image vectors

        """
        import numpy as np

        self.vectors = np.load(self.config["image_vectorizer"]["vectors_path"])
        self.next(self.create_index)

    @step
    def create_index(self):
        """
        Create ann index

        """
        from image_vectorizer.cbir_functions import build_ann_model

        build_ann_model(
            self.vectors,
            self.config["ann_model"]["init"],
            self.config["ann_model"]["index_time"],
            self.config["ann_model"]["ann_model_path"],
        )
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    DownloadingPipeline()
