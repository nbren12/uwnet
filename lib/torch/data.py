"""This module contains routines for handling the intermediate representation
of the data. The routines in this module all have a dict `data` as the first
argument. This dict should have the same structure as produced by `prepare_data`.

Training.py uses the following interface:

to_dataset
to_constants ...done
to_scaler ...done
to_dynamics_loss

the useful parts of ..data are

compute_weighted_scale : computes the scales
prepare_array : transposes the data correctly
the material derivative part of inputs_and_forcings



See Also
--------
lib.data.prepare_data

"""
import logging

import attr
import numpy as np
import torch
from toolz import valmap, curry
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

import xarray as xr

from ..thermo import layer_mass, liquid_water_temperature
from .datasets import DictDataset, WindowedData
from .loss import dynamic_loss
from .normalization import scaler

logger = logging.getLogger(__name__)



def _compute_weighted_scale(x, layer_mass):
    sample_dims = tuple(range(x.ndim - 1))
    sig2 = x.var(axis=sample_dims)

    assert sig2.shape == layer_mass.shape

    avg_sig2 = (sig2 * layer_mass / layer_mass.sum()).sum()
    return np.sqrt(avg_sig2)


def _loss_weight(x, layer_mass):
    """Compute MSE loss weight for numpy array

    Parameters
    ----------
    x : (..., z)
    layer_mass : (z, )
    """
    scale = _compute_weighted_scale(x, layer_mass)
    return layer_mass / scale**2


def prepare_array(x):
    output_dims = [dim for dim in ['time', 'y', 'x', 'z'] if dim in x.dims]
    return x.transpose(*output_dims).values.astype(np.float32)


@curry
def collate_fn(constants, seq):
    collated = default_collate(seq)
    d = _transpose_vars(collated)
    d['constant'] = constants
    return d


def _transpose_vars(data):
    if torch.is_tensor(data):
        return data.transpose(0, 1)
    elif isinstance(data, dict):
        return {
            key: _transpose_vars(val)
            for key, val in data.items()
        }


@attr.s
class TrainingData(object):
    qt = attr.ib()
    sl = attr.ib()
    FQT = attr.ib()
    FSL = attr.ib()
    SHF = attr.ib()
    LHF = attr.ib()
    QRAD = attr.ib()
    SOLIN = attr.ib()
    layer_mass = attr.ib()
    z = attr.ib()
    p = attr.ib()

    def constants(self):
        return {
            'w': self.layer_mass,
            'z': self.z
        }

    def prognostic(self):
        return {
            'qt': self.qt,
            'sl': self.sl,
        }

    def forcing(self):
        # create forcing dictionary
        auxiliary_names = ('QRAD', 'LHF', 'SHF', 'SOLIN')
        forcing_variables = {
            'sl': self.FSL,
            'qt': self.FQT,
        }
        for name in auxiliary_names:
            forcing_variables[name] = getattr(self, name)

        return forcing_variables

    def scaler(self):
        # compute mean and stddev
        # this is an error, std does not work like this
        means = {}
        scales = {}
        input_vars = {'qt': self.qt, 'sl': self.sl}

        for key, X in input_vars.items():
            m = X.shape[-1]
            mu = X.reshape((-1, m)).mean(axis=0)
            sig = X.reshape((-1, m)).std(axis=0)

            # convert to torch
            mu, sig = [torch.from_numpy(np.squeeze(x)) for x in [mu, sig]]

            # take average in vertical direction
            sig = torch.mean(sig)

            means[key] = mu
            scales[key] = sig

        return scaler(scales, means)

    def dynamic_loss(self, **kwargs):
        weights = {
            'qt': _loss_weight(self.qt, self.layer_mass),
            'sl': _loss_weight(self.sl, self.layer_mass),
        }
        weights = valmap(torch.from_numpy, weights)
        return dynamic_loss(weights=weights, **kwargs)

    def get_num_features(self):
        return len(self.p) * 2

    def torch_dataset(self, window_size):

        # create prognostic dict
        prognostic_variables = {'sl': self.sl, 'qt': self.qt}

        # create forcing dictionary
        auxiliary_names = ('QRAD', 'LHF', 'SHF', 'SOLIN')
        forcing_variables = {
            'sl': self.FSL,
            'qt': self.FQT,
        }
        for name in auxiliary_names:
            forcing_variables[name] = getattr(self, name)

        # turn these into windowed datasets
        X = DictDataset({
            key: WindowedData(val, window_size)
            for key, val in prognostic_variables.items()
        })
        G = DictDataset({
            key: WindowedData(val, window_size)
            for key, val in forcing_variables.items()
        })

        return DictDataset({'prognostic': X, 'forcing': G})

    def get_loader(self, window_size, num_samples=None, batch_size=None,
                   **kwargs):
        """Return the whole dataset as a batch for input to ForcedStepper

        Yields
        -------
        batch : dict
            contains keys 'constant', 'prognostic', 'forcing'.

        """
        dataset = self.torch_dataset(window_size)


        # Create training data loaders
        if num_samples:
            logger.info(f"Using boostrap sample of {num_samples}.")
            inds = np.random.choice(len(dataset), num_samples,
                                    replace=False)
            kwargs['sampler'] = SubsetRandomSampler(inds)
        else:
            logger.info(f"Using full training dataset")
            num_samples = len(dataset)

        if batch_size:
            kwargs['batch_size'] = batch_size
        else:
            kwargs['batch_size'] = num_samples

        constants = valmap(torch.from_numpy, self.constants())
        return DataLoader(dataset, collate_fn=collate_fn(constants),
                          **kwargs)


    def from_files(paths, post=None):
        """Create TrainingData from filenames

        Examples
        --------
        >>> TrainingData.from_files(default='all.nc', p='p.nc')
        """
        init_kwargs = {}
        for key in paths:
            logger.debug(f"Loading {paths[key]}")
            init_kwargs[key] = xr.open_dataset(
                paths[key], chunks={
                    'time': 10
                })[key]

        # compute layer mass from stat file
        if 'RHO' in paths:
            rho = init_kwargs.pop('RHO')[0]
            rhodz = layer_mass(rho)

            init_kwargs['layer_mass'] = rhodz
            init_kwargs['z'] = rhodz.z

        # compute prognostics variables from TABS
        if 'TABS' in paths:
            TABS = init_kwargs.pop('TABS')
            QV = init_kwargs.pop('QV')
            QN = init_kwargs.pop('QN', 0.0)
            QP = init_kwargs.pop('QP', 0.0)

            sl = liquid_water_temperature(TABS, QN, QP)
            qt = QV + QN

            init_kwargs['sl'] = sl
            init_kwargs['qt'] = qt

        if post:
            init_kwargs = valmap(post, init_kwargs)

        # process all files into numpy arrays
        init_kwargs = {key: prepare_array(x) for key, x in init_kwargs.items()}

        return TrainingData(**init_kwargs)

    @classmethod
    def from_var_files(cls, files, **kwargs):
        paths = {}
        for path, variables in files:
            paths.update({key: path for key in variables})

        return cls.from_files(paths, **kwargs)

    def save(self, path):
        torch.save(attr.asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path))

    @property
    def nbytes(self):
        total = 0
        for val in attr.astuple(self):
            total += val.nbytes
        return total

    @property
    def shape(self):
        return self.FQT.shape
