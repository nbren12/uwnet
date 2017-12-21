"""Variational Encoder/Decoder models for convection


The inputs are [qt and sl] and the outputs are Q1 and Q2.


Notes
-----
This is based on the Variational Auto-encoder example from pyro_.

.. _pyro: <https://github.com/uber/pyro/blob/dev/examples/vae.py>

Examples
--------

Run this on the command line like this::
    python  encdec.py ../../data/ml/ngaqua/data.pkl \
              --max-iter 1000 -n 4 --learning-rate=1e-3
"""
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

from tqdm import tqdm

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_ones, ng_zeros
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from sklearn.externals import joblib

import xarray as xr

from lib.util import mat_to_xarray
from lib.models import mean_squared_error, weighted_r2_score

fudge = 1e-7


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
# This class will actually be used for both encoding and decoding
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        # setup the three linear transformations used
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, out_dim)
        self.fc22 = nn.Linear(hidden_dim, out_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()

    def forward(self, x):
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self,
                 in_dim,
                 out_dim,
                 z_dim=10,
                 hidden_dim=100,
                 use_cuda=False):
        super(VAE, self).__init__()

        # create the encoder and decoder networks
        self.encoder = Encoder(in_dim, hidden_dim, z_dim)
        self.decoder = Encoder(z_dim, hidden_dim, out_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda

        self.z_dim = z_dim
        self.out_dim = out_dim
        self.in_dim = in_dim

    # define the model p(x|z)p(z)
    def model(self, data):
        """the inputs and outputs are concanated together

        so I need to extract them see the `Bayesian linear regression example
        <http://pyro.ai/examples/bayesian_regression.html>`_.

        """

        x = data[:, :self.in_dim]
        y = data[:, self.in_dim:]

        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        # setup hyperparameters for prior p(z)
        # the type_as ensures we get cuda Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=y.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=y.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        # decode the latent code z
        mu_img, sig_img = self.decoder.forward(z)
        # score against actual images
        return pyro.observe("obs", dist.normal, y, mu_img, sig_img)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, data):

        x = data[:, :self.in_dim]
        y = data[:, self.in_dim:]

        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder.forward(x)
        # sample the latent code z
        return pyro.sample("latent", dist.normal, z_mu, z_sigma)


class PklDataset(Dataset):
    """PyTorch dataset for data stored in .pkl format"""

    def __init__(self, filename, key='train'):
        "docstring"
        self.data = joblib.load(filename)

        self.key = key
        # make a standardizer
        self.scaler = make_pipeline(VarianceThreshold(.001), StandardScaler())
        self.scaler.fit(self.data['train'][0])

        self.input_data = self.scaler.transform(self.data[self.key][0])
        # get underlying data from xarray for output
        self.output_data = self.data[self.key][1]

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return (self.input_data[idx], self.output_data.data[idx])

    @property
    def ycoords(self):
        return self.output_data.coords


def get_data_loaders(datafile):
    """Create training and testing data loaders for data stored in a pickle file"""
    train_data = PklDataset(datafile, key='train')
    test_data = PklDataset(datafile, key='test')

    return [
        DataLoader(data, batch_size=100, shuffle=True)
        for data in [train_data, test_data]
    ]


def evaluate_model_fit(vae, datafile):

    # Load full dataset
    train_data = PklDataset(datafile, key='train')
    x, _ = train_data[:]
    x_torch = Variable(torch.FloatTensor(x))

    mu, sig = vae.encoder.forward(x_torch)
    pred_mu, pred_sig = vae.decoder.forward(mu)

    _, w = train_data.data['w']

    # just use the point estimate as the prediction
    y = train_data.output_data
    pred = xr.DataArray(pred_mu.data.numpy(), train_data.ycoords)
    train_score = weighted_r2_score(y, pred, w)

    # compute MEAN_SQUARED_ERROR and variance
    mean_square_error = mean_squared_error(y, pred, axis=0).unstack("features")
    variance = mean_squared_error(y, y.mean("samples"), axis=0).unstack("features")

    train_score = weighted_r2_score(y, pred, w)
    print("Training score", train_score)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('data', type=str, help='data stored in pickle format')
    parser.add_argument(
        '-n',
        '--num-epochs',
        default=1,
        type=int,
        help='number of training epochs')
    parser.add_argument(
        '--max-iter',
        default=1000,
        type=int,
        help='maximum number of iterations')
    parser.add_argument(
        '-tf',
        '--test-frequency',
        default=5,
        type=int,
        help='how often we evaluate the test set')
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1.0e-3,
        type=float,
        help='learning rate')
    parser.add_argument(
        '-b1',
        '--beta1',
        default=0.95,
        type=float,
        help='beta1 adam hyperparameter')
    parser.add_argument(
        '--cuda',
        action='store_true',
        default=False,
        help='whether to use cuda')
    parser.add_argument(
        '-i-tsne',
        '--tsne_iter',
        default=100,
        type=int,
        help='epoch when tsne visualization runs')
    args = parser.parse_args()

    # TODO change this section
    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = get_data_loaders(args.data)

    # get size of input and output dimensions
    x, y = next(iter(train_loader))
    in_dim, out_dim = x.shape[1], y.shape[1]

    # setup the VAE
    vae = VAE(in_dim, out_dim, use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

    train_elbo = []
    test_elbo = []
    # training loop
    total_iter = 0

    generator = iter(train_loader)

    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for k, (x, y) in tqdm(
                enumerate(generator),
                total=min(len(train_loader), args.max_iter)):

            # I need to cast these types manually here
            x = x.float()
            y = y.float()

            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            # wrap the mini-batch in a PyTorch Variable
            data = torch.cat((x, y), 1)
            data = Variable(data)
            # do ELBO gradient and accumulate loss
            l = svi.loss(vae.model, vae.guide, data)
            if np.isnan(l):
                raise FloatingPointError("test loss is nan")

            epoch_loss += svi.step(data)

            if np.isnan(epoch_loss):
                raise FloatingPointError("test loss is nan")

            if k > args.max_iter:
                break

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" %
              (epoch, total_epoch_loss_train))

        # if epoch % args.test_frequency == 0:
        #     # initialize loss accumulator
        #     test_loss = 0.
        #     # compute the loss over the entire test set
        #     for i, (x, y) in enumerate(test_loader):

        #         # I need to cast these types manually here
        #         x = x.float()
        #         y = y.float()

        #         # if on GPU put mini-batch into CUDA memory
        #         if args.cuda:
        #             x = x.cuda()
        #             y = y.cuda()

        #         # wrap the mini-batch in a PyTorch Variable
        #         data = torch.cat((x, y), 1)
        #         data = Variable(data)

        #     # report test diagnostics
        #     normalizer_test = len(test_loader.dataset)
        #     total_epoch_loss_test = test_loss / normalizer_test
        #     test_elbo.append(total_epoch_loss_test)
        #     print("[epoch %03d] average test loss: %.4f" %
        #           (epoch, total_epoch_loss_test))

    evaluate_model_fit(vae, args.data)
    return vae


if __name__ == '__main__':
    model = main()
