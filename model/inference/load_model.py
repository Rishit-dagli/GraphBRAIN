import wget
import hashlib
import tensorflow as tf

def _download() -> str:
    url = "http://store.rishit.tech/model.tar.gz"
    filename = wget.download(url)
    return filename


def _check_md5(filename: str) -> bool:
    expected_md5 = "fcd1cf90bf9f6a87328a5e1b18b1a637"
    with open(filename, "rb") as f:
        actual_md5 = hashlib.md5(f.read()).hexdigest()
    return expected_md5 == actual_md5


def download_model() -> str:
    print("Downloading Model...")
    filename = _download()
    if _check_md5(filename):
        print("Downloaded Model")
        return filename
    else:
        raise ValueError("MD5 Checksum failed")

def load_model(filename = None) -> tf.keras.Model:
    if filename is None:
        filename = download_model()
    model = tf.keras.models.load_model(filename)
    return model