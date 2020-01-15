import fsspec
import pickle
import logging


width = 6


def load_pickle_from_url(url):
    logging.info(f"Opening {url}")
    openfile = fsspec.open(url, mode="rb")
    with openfile as f:
        s = f.read()
    return pickle.loads(s)
