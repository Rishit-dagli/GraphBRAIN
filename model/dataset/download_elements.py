"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function for downloading the periodic table dataset.

The application has the following functions:


Copyright and Usage Information
===============================
This file is provided solely for the personal and private use of TAs, instructors and its author(s). All forms of
distribution of this code, whether as given or with any changes, are expressly prohibited.

This file is Copyright (c) 2023 by Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde."""

import wget
import hashlib
import os
import python_ta as pyta


def _download() -> str:
    """Download the periodic table dataset from the internet.

    Returns:
        the name of the downloaded file.
    """
    url = "http://store.rishit.tech/periodictable.csv"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Check the MD5 checksum of the downloaded file.

    Arguments:
        filename: the name of the downloaded file.

    Returns:
        True if the MD5 checksum matches, False otherwise.
    """
    expected_md5 = "294354a8981426b9d83ea05d65ea0d1b"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
        print(actual_md5)
    return expected_md5 == actual_md5


def download_periodic() -> str:
    """Download the periodic table dataset if it is not already downloaded.

    Returns:
        the name of the downloaded file.
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
        print("Periodic dataset already downloaded")
        return "periodictable.csv"


pyta.check_all(
    config={
        "extra-imports": ["wget", "hashlib", "os", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output10.txt",
)
