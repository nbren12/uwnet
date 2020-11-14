import fsspec
import pickle
import logging
import numpy as np


width = 6


def load_pickle_from_url(url):
    logging.info(f"Opening {url}")
    openfile = fsspec.open(url, mode="rb")
    with openfile as f:
        s = f.read()
    return pickle.loads(s)


def add_wavelength_colorbar(fig, im, **kwargs):

    wave_length = np.array([10, 100, 200, 300, 500, 1000, 10000])
    tick_locations = 2 * np.pi / wave_length / 1e3

    cb = fig.colorbar(im, ticks=tick_locations, **kwargs)
    cb.ax.set_yticklabels(wave_length)
    cb.set_label('Wavelength (km)')
    return cb
