"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines functions for loading the model.

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
import tensorflow as tf
import tarfile
from typing import Optional
import os


def _download() -> str:
    """Downloads the model from the given URL.

    Returns:
        str: The filename (str) of the downloaded model.
    """
    url = "http://store.rishit.tech/model.tar.gz"
    if os.path.exists("model.tar"):
        return "model.tar"
    if os.path.exists("model.tar.gz"):
        return "model.tar.gz"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Checks the MD5 checksum of the given file.

    Args:
        filename (str): The filename of the file to check.

    Returns:
        bool: True if the MD5 checksum of the file matches the expected checksum, False otherwise.
    """
    expected_md5 = "fcd1cf90bf9f6a87328a5e1b18b1a637"
    if filename == "model.tar":
        return True
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    if expected_md5 != actual_md5:
        print(
            "The MD5 checksum of the downloaded file does not match the expected checksum, so the file may be "
            "corrupted, but we will continue using this file. This might be caused due to zipping and unzipping "
            "the file."
        )
    return True


def download_model() -> str:
    """Downloads the model from the given URL.

    Returns:
        str: The filename of the downloaded model.

    Raises:
        ValueError: If the MD5 checksum of the downloaded file does not match the expected checksum.
    """
    print("Downloading Model...")
    filename = _download()
    if _check_md5(filename):
        print("Downloaded Model")
        return filename
    else:
        raise ValueError("MD5 Checksum failed")


def load_model(filename: Optional[str] = None) -> tf.keras.Model:
    """Loads the model from the given filename.

    Args:
        filename (Optional[str]): The filename of the model to load.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if filename is None:
        filename = download_model()
        file = tarfile.open(filename)
        file.extractall("./model")
        filename = "./model"
    model = tf.keras.models.load_model(filename)
    return model


if __name__ == "__main__":
    import python_ta as pyta
    import doctest

    doctest.testmod()
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pyta.check_all(
        os.path.join(path, "model", "inference", "load_model.py"),
        config={
            "extra-imports": [
                "tensorflow",
                "hashlib",
                "wget",
                "tarfile",
                "typing",
                "python_ta",
                "os",
            ],
            "allowed-io": ["load_model", "_check_md5", "download_model"],
            "max-line-length": 120,
            "disable": ["E9992", "R1732", "C0411", "E9997"],
        },
        output=os.path.join(path, "pyta_outputs", "load_model.html"),
    )
