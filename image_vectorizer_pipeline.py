import os
import pathlib
from metaflow import FlowSpec, step, Parameter


class VectorizingPipeline(FlowSpec):
    FILE_DIR = pathlib.Path(__file__).parent.absolute()

    PICTURES_PATH = Parameter(
        "pictures_path", default=os.path.join(FILE_DIR, "pictures")
    )

    FILELIST_FILENAME = Parameter(
        "filelist_filename", default=os.path.join(FILE_DIR, "file_list.txt")
    )

    VECTORS_FILENAME = Parameter(
        "vectors_filename", default=os.path.join(FILE_DIR, "vectors.npy")
    )

    @step
    def start(self):
        """
        Just to kick things off
        """
        self.next(self.vectorize_images)

    @step
    def vectorize_images(self):
        """
        Vectorizes downloaded images and saves numpy array with all vectors

        """
        import os
        import numpy as np
        from image_vectorizer.image_vectorizing_functions import generate_vectors

        print("VECTORIZING IMAGES")
        file_list, vectors = generate_vectors(
            os.path.join(self.FILE_DIR, self.PICTURES_PATH)
        )
        with open(os.path.join(self.FILE_DIR, self.FILELIST_FILENAME), "w") as f:
            f.write("\n".join(file_list))
        np.save(os.path.join(self.FILE_DIR, self.VECTORS_FILENAME), vectors)
        self.next(self.end)

    @step
    def end(self):
        """
        All done

        """
        print("DONE")


if __name__ == "__main__":
    VectorizingPipeline()
