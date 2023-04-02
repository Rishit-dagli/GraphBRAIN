"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines  function for downloading the periodic table dataset.

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
    """Downloads the periodic table dataset from the internet.

    Returns:
        A string representing the name of the downloaded file.
    """
    url = "http://store.rishit.tech/periodictable.csv"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Checks the MD5 checksum of the downloaded file.

    Args:
        filename (str): A string representing the name of the downloaded file.

    Returns:
        bool: A boolean value indicating whether the MD5 checksum matches or not.
    """
    expected_md5 = "294354a8981426b9d83ea05d65ea0d1b"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
        print(actual_md5)
    return expected_md5 == actual_md5


def download_periodic() -> str:
    """Downloads the periodic table dataset if it is not already downloaded.

    Returns:
        str: A string representing the name of the downloaded file.

    Raises:
        ValueError: If the MD5 checksum of the downloaded dataset does not match.
    """
    if not os.path.exists("periodictable.csv"):
        print("Downloading Periodic Table dataset...")
        filename = _download()
        if _check_md5(filename):
            print("Downloaded Periodic dataset")
            return filename
        else:
            raise ValueError("MD5 Checksum failed")
    else:
        return "periodictable.csv"


if __name__ == '__main__':
    import python_ta as pyta
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyta.check_all(
        os.path.join(path, "model", "dataset", "download_elements.py"),
        config={
            "extra-imports": ["wget", "hashlib", "os", "python_ta"],
            "allowed-io": ["_check_md5", "download_periodic"],
            "max-line-length": 120,
            "disable": ["E9992", "C0411", "E9997"],
        },
        output=os.path.join(path, "pyta_outputs", "download_elements.html"),
    )
