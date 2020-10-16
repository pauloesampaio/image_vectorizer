import pymongo
import os
import requests
import pathlib
from concurrent.futures import ThreadPoolExecutor


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


def _check_if_exists(path_to_check, create=False):
    if not os.path.exists(path_to_check):
        if create:
            pathlib.Path(path_to_check).mkdir(parents=True, exist_ok=True)
            return print(f"{path_to_check} created")
        else:
            return None


def _download_image(product_tuple):
    product_hash, image_url, download_dir, db_name, retailer = product_tuple
    file_path = os.path.join(
        download_dir, db_name, retailer, f"{product_hash}_{image_url.split('/')[-1]}"
    )
    _check_if_exists(os.path.dirname(file_path), create=True)
    try:
        resp = requests.get(image_url, stream=True)
        with open(file_path, "wb") as f:
            f.write(resp.content)
    except IOError:
        return print(f"Error saving {file_path}")


def download_images(download_list):
    with ThreadPoolExecutor() as executor:
        executor.map(_download_image, download_list)
    return print("Downloads done")
