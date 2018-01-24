"""Module for denoising the forcings using auto-encoders
"""
import numpy as np
import xarray as xr
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential


def autoencoder(n, m):

    model = Sequential([
        Dense(34, input_shape=(n, )),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(34),
        Activation('relu'),
        Dense(m)
    ])

    model.compile(optimizer='rmsprop', loss='mse')

    return model


def train_denoiser(x, n_remove_top=10):
    x_old = x
    x = x.values
    mu = x.mean(axis=0)
    sig2 = ((x - mu)**2).mean()
    sig = np.sqrt(sig2)
    dm = (x - mu) / sig

    m = x.shape[1]

    model = autoencoder(m - n_remove_top, m)
    model.fit(dm[:, :-n_remove_top], dm, epochs=1)

    def denoise(x):
        x_transformed = (x - mu) / sig
        pred = model.predict(x_transformed[:, :-n_remove_top])
        return pred * sig + mu

    return denoise


def denoise(x, **kwargs):
    stacked = x.stack(samples=['time', 'x', 'y']).transpose('samples', 'z')
    denoiser = train_denoiser(stacked, **kwargs)
    denoised = denoiser(stacked)
    return xr.DataArray(denoised, coords=stacked.coords).unstack('samples')
