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

        self.next(self.get_paths)

    @step
    def get_paths(self):
        """
        Get image paths
        """
        from image_vectorizer.utils import get_paths_dataframe

        self.paths_dataframe = get_paths_dataframe(
            self.config["pictures_path"],
            self.config["infer_classes"],
        )

        self.next(self.vectorize_images)

    @step
    def vectorize_images(self):
        """
        Vectorizes downloaded images and saves numpy array with all vectors

        """
        from image_vectorizer.image_vectorizing_functions import generate_vectors

        self.vectors = generate_vectors(
            self.paths_dataframe,
            self.config["infer_classes"],
            self.config["reduce_vector_dimensionality"],
        )
        self.next(self.save_vectors)

    @step
    def save_vectors(self):
        """
        Save vectors and file list

        """
        import os
        from image_vectorizer.utils import save_array

        save_array(
            os.path.join(self.FILE_DIR, self.config["vectors_path"]),
            self.vectors,
        )

        self.paths_dataframe.to_csv(
            os.path.join(self.FILE_DIR, self.config["file_list_path"]), index=False
        )
        self.next(self.calculate_tsne)

    @step
    def calculate_tsne(self):
        """
        Calculate TSNE coordinates

        """
        from sklearn.manifold import TSNE
        from image_vectorizer.utils import save_array

        if self.config["generate_tsne"]:
            tsne = TSNE(random_state=12345, verbose=2, n_jobs=-1)
            xy = tsne.fit_transform(self.vectors)
            save_array(self.config["tsne_path"], xy)

        self.next(self.end)

    @step
    def end(self):
        """
        All done

        """
        print("DONE")


if __name__ == "__main__":
    VectorizingPipeline()
