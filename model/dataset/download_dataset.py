"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for downloading the BBBP dataset.

Copyright and Usage Information
===============================
Copyright 2023 Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import wget
import hashlib
import os


def _download() -> str:
    """Downloads the BBBP dataset from the Rishit's website.

    Returns:
        str: The filename of the downloaded dataset as a string.
    """
    try:
        url = "http://store.rishit.tech/BBBP.csv"
        filename = wget.download(url)
    except:
        url = "https://github.com/Rishit-dagli/csc111-project/releases/download/weights/BBBP.csv"
        filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Checks the MD5 checksum of the downloaded dataset.

    Args:
        filename (str): A string representing the filename of the downloaded dataset.

    Returns:
        bool: A boolean value indicating whether the MD5 checksum matches or not.
    """
    expected_md5 = "66286cb9e6b148bd75d80c870df580fb"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    if expected_md5 != actual_md5:
        print(
            "The MD5 checksum of the downloaded file does not match the expected checksum, so the file may be "
            "corrupted, but we will continue using this file. This might be caused due to zipping and unzipping "
            "the file."
        )
    return True


def download_dataset() -> str:
    """Downloads the BBBP dataset if it is not already downloaded.

    Returns:
        str: A string representing the filename of the downloaded dataset.

    Raises:
        ValueError: If the MD5 checksum of the downloaded dataset does not match.
    """
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


if __name__ == "__main__":
    import python_ta as pyta

    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyta.check_all(
        os.path.join(path, "model", "dataset", "download_dataset.py"),
        config={
            "extra-imports": ["wget", "hashlib", "os", "python_ta"],
            "allowed-io": ["_check_md5", "download_dataset"],
            "max-line-length": 120,
            "disable": ["E9992", "C0411", "E9997"],
        },
        output=os.path.join(path, "pyta_outputs", "download_dataset.html"),
    )
