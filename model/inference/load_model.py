"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for loading the model.

The application has the following functions:
_download() -> srt: Download the model from the given URL.

_check_md5(filename: str) -> bool: Check the MD5 checksum of the given file.

download_model() -> str: Download the model from the given URL.

load_model(filename=None) -> tf.keras.Model: Load the model from the given filename.

Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import wget
import hashlib
import tensorflow as tf
import tarfile
import python_ta as pyta


def _download() -> str:
    """Download the model from the given URL.

    Arguments:
        None

    Returns:
        The filename of the downloaded model.
    """
    url = "http://store.rishit.tech/model.tar.gz"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Check the MD5 checksum of the given file.

    Arguments:
        filename: The filename of the file to check.

    Returns:
        True if the MD5 checksum of the file matches the expected checksum, False otherwise.
    """
    expected_md5 = "fcd1cf90bf9f6a87328a5e1b18b1a637"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    return expected_md5 == actual_md5


def download_model() -> str:
    """Download the model from the given URL.

    Arguments:
        None

    Returns:
        The filename of the downloaded model.

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


def load_model(filename=None) -> tf.keras.Model:
    """Load the model from the given filename.

    Arguments:
        filename: The filename of the model to load.

    Returns:
        The loaded model.
    """
    if filename is None:
        filename = download_model()
        file = tarfile.open(filename)
        file.extractall("./model")
        filename = "./model"
    model = tf.keras.models.load_model(filename)
    return model


pyta.check_all(
    config={
        "extra-imports": ["tensorflow", "hashlib", "wget", "tarfile", "python_ta"],
        "allowed-io": ["load_model", "_check_md5"],
        "max-line-length": 120,
    },
    output="pyta_output7.txt",
)
