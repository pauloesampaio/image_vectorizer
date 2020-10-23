from metaflow import FlowSpec, step


class VectorizingPipeline(FlowSpec):
    @step
    def start(self):
        """
        Loading config
        """
        import pathlib
        from image_vectorizer.utils import load_config

        self.FILE_DIR = pathlib.Path(__file__).parent.absolute()
        self.config = load_config()

        self.next(self.vectorize_images)

    @step
    def vectorize_images(self):
        """
        Vectorizes downloaded images and saves numpy array with all vectors

        """
        import os
        from image_vectorizer.image_vectorizing_functions import generate_vectors

        print("VECTORIZING IMAGES")
        self.file_list, self.vectors = generate_vectors(
            os.path.join(
                self.FILE_DIR, self.config["pictures_downloader"]["pictures_path"]
            )
        )
        self.next(self.save_vectors)

    @step
    def save_vectors(self):
        """
        Save vectors and file list

        """
        import os
        import numpy as np

        with open(
            os.path.join(
                self.FILE_DIR, self.config["image_vectorizer"]["file_list_path"]
            ),
            "w",
        ) as f:
            f.write("\n".join(self.file_list))
        np.save(
            os.path.join(
                self.FILE_DIR, self.config["image_vectorizer"]["vectors_path"]
            ),
            self.vectors,
        )
        self.next(self.end)

    @step
    def end(self):
        """
        All done

        """
        print("DONE")


if __name__ == "__main__":
    VectorizingPipeline()
