import os
import pathlib
from metaflow import FlowSpec, step, Parameter


class DownloadingPipeline(FlowSpec):
    FILE_DIR = pathlib.Path(__file__).parent.absolute()

    PICTURES_PATH = Parameter(
        "pictures_path", default=os.path.join(FILE_DIR, "pictures")
    )

    CREDENTIALS_PATH = Parameter(
        "credentials_path",
        default=os.path.join(FILE_DIR, "credentials"),
    )

    @step
    def start(self):
        """
        Load the credentials file.

        """
        import os
        import json

        with open(
            os.path.join(".", self.CREDENTIALS_PATH, "credentials.json"), "r"
        ) as f:
            self.credentials = json.load(f).get("mongo_db")
        self.next(self.query_db)

    @step
    def query_db(self):
        """
        Query DB to get images url to download

        """
        from image_vectorizer.images_download_functions import query_db

        self.download_list = query_db(
            credentials=self.credentials, download_dir=self.PICTURES_PATH
        )
        print(f"Found {len(self.download_list)} pictures on the database")
        self.next(self.download_pictures)

    @step
    def download_pictures(self):
        """
        Download images using multiprocess to speed it up

        """
        from image_vectorizer.images_download_functions import download_images

        download_images(self.download_list)
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    DownloadingPipeline()
