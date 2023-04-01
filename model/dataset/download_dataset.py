import wget
import hashlib
import os


def _download() -> str:
    url = "http://store.rishit.tech/BBBP.csv"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    expected_md5 = "66286cb9e6b148bd75d80c870df580fb"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    return expected_md5 == actual_md5


def download_dataset() -> str:
    if not os.path.exists("BBBP.csv"):
        print("Downloading BBBP dataset...")
        filename = _download()
        if _check_md5(filename):
            print("Downloaded BBBP dataset")
            return filename
        else:
            raise ValueError("MD5 Checksum failed")
    else:
        print("BBBP dataset already downloaded")
        return "BBBP.csv"
