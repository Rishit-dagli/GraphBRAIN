import wget
import hashlib


def _download() -> str:
    url = "http://store.rishit.tech/periodictable.csv"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    expected_md5 = "294354a8981426b9d83ea05d65ea0d1b"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
        print(actual_md5)
    return expected_md5 == actual_md5


def download_periodic() -> str:
    print("Downloading Periodic Table dataset...")
    filename = _download()
    if _check_md5(filename):
        print("Downloaded Periodic dataset")
        return filename
    else:
        raise ValueError("MD5 Checksum failed")
