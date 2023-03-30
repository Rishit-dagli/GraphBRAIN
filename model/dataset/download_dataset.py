import wget
import hashlib
import os
import python_ta as pyta


def _download() -> str:
    """Download the BBBP dataset from the Rishit's website.

    Returns:
        The filename of the downloaded dataset.
    """
    url = "http://store.rishit.tech/BBBP.csv"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    """Check the MD5 checksum of the downloaded dataset.

    Returns:
        True if the MD5 checksum matches, False otherwise.
    """
    expected_md5 = "66286cb9e6b148bd75d80c870df580fb"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    return expected_md5 == actual_md5


def download_dataset() -> str:
    """Download the BBBP dataset if it is not already downloaded.

    Returns:
        The filename of the downloaded dataset.
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


pyta.check_all(
    config={
        "extra-imports": ["wget", "hashlib", "os", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output11.txt",
)
