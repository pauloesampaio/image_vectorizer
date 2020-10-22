import pymongo
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from .utils import check_if_exists, image_loader


def query_db(credentials, download_dir):
    connector = (
        f"mongodb+srv://{credentials['user']}:{credentials['password']}"
        f"@{credentials['cluster']}.qe0ku.gcp.mongodb.net/{credentials['database']}?retryWrites=true&w=majority"
    )
    client = pymongo.MongoClient(connector)
    db = client[credentials["database"]]
    docs = db[credentials["collection"]].find({})
    download_list = []
    for doc in docs:
        download_list = download_list + [
            (
                doc["_id"],
                url,
                download_dir,
                credentials["database"],
                doc["product_retailer"],
            )
            for url in doc["product_images"]
        ]
    return download_list


def _download_image_multithread_helper(product_tuple):
    product_hash, image_url, download_dir, db_name, retailer = product_tuple
    file_path = os.path.join(
        download_dir, db_name, retailer, f"{product_hash}_{image_url.split('/')[-1]}"
    )
    check_if_exists(os.path.dirname(file_path), create=True)
    try:
        image = image_loader(image_url, remote=True)
        cv2.imwrite(file_path, image)
    except IOError:
        return print(f"Error saving {file_path}")


def download_image_multithread(download_list):
    with ThreadPoolExecutor() as executor:
        executor.map(_download_image_multithread_helper, download_list)
    return print("Downloads done")
