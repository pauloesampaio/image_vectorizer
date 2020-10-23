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
        self.next(self.get_credentials)

    @step
    def get_credentials(self):
        """
        Load the credentials file.

        """
        import json

        with open(
            os.path.join(
                self.FILE_DIR, self.config["pictures_downloader"]["credentials_path"]
            ),
            "r",
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
            credentials=self.credentials,
            download_dir=self.config["pictures_downloader"]["pictures_path"],
        )
        print(f"Found {len(self.download_list)} pictures on the database")
        self.next(self.download_pictures)

    @step
    def download_pictures(self):
        """
        Download images using multiprocess to speed it up

        """
        from image_vectorizer.images_download_functions import (
            download_image_multithread,
        )

        download_image_multithread(self.download_list)
        self.next(self.end)

    @step
    def end(self):
        """
        All done!

        """
        print("DONE")


if __name__ == "__main__":
    DownloadingPipeline()
