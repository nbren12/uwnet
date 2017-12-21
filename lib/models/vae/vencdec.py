"""Same as encdec.py but uses `edward`



"""
#!/usr/bin/env python
"""Variational auto-encoder for MNIST data.

References
----------
http://edwardlib.org/tutorials/decoder
http://edwardlib.org/tutorials/inference-networks
"""

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Bernoulli, Normal
from edward.util import Progbar
from keras.layers import Dense
from keras import Sequential
from observations import mnist
from scipy.misc import imsave

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.metrics import r2_score


class PklDataset(object):
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
        self.output_data = self.data[self.key][1].data

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return (self.input_data[idx], self.output_data[idx])


def generator(dataset, batch_size=40):
    offset = batch_size
    n = len(dataset)
    while True:
        sl = slice(offset, min(n, offset+batch_size))
        yield dataset[sl]


dataset = PklDataset("../../data/ml/ngaqua/data.pkl") 
ed.set_seed(42)

data_dir = "/tmp/data"
out_dir = "/tmp/out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
M = 100  # batch size during training
d = 10  # latent dimension
nhid = 40 # hidden dimension size


# get number of input features
n_in = dataset[0][0].shape[0]
n_out = dataset[0][1].shape[0]

# placeholders
x_ph = tf.placeholder(tf.float32, [None, n_in])
y_ph = tf.placeholder(tf.float32, [None, n_out])
z_ph = tf.placeholder(tf.float32, [None, d])

# MODEL
# Define a subgraph of the full model, corresponding to a minibatch of
# size M.

# we need handles to these models
# for later evaluation
sig_model = Sequential()
sig_model.add(Dense(nhid, activation='relu', input_shape=(d,)))
sig_model.add(Dense(n_out, activation='softplus'))

mu_model = Sequential()
mu_model.add(Dense(nhid, activation='relu', input_shape=(d,)))
mu_model.add(Dense(n_out))

z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
y = Normal(loc=mu_model(z.value()), scale=sig_model(z.value()))


# INFERENCE
# Define a subgraph of the variational model, corresponding to a
# minibatch of size M.
hidden = Dense(100, activation='relu')(tf.cast(x_ph, tf.float32))
qz = Normal(loc=Dense(d)(hidden),
            scale=Dense(d, activation='softplus')(hidden))

# Bind p(y, z) and q(z | y) to the same TensorFlow placeholder for y.
inference = ed.KLqp({z: qz}, data={y: y_ph})

# point estimate model
mu_from_x = mu_model(qz.loc)
sig_from_x = sig_model(qz.loc)

# stochastic model for Q1 and Q2
y_stoch = Normal(loc=mu_model(qz.value()),
                 scale=sig_model(qz.value()))

# i need to pass a scope argument here
# i wasted hours on this crap
y_point = ed.copy(y, {z: qz.loc}, scope='point')


optimizer = tf.train.RMSPropOptimizer(0.001, epsilon=1.0)
inference.initialize(optimizer=optimizer)

tf.global_variables_initializer().run()

n_epoch = 10
n_iter_per_epoch = 2000

x_train_generator = generator(dataset, batch_size=M)
pbar = Progbar(n_iter_per_epoch)

for epoch in range(1, n_epoch + 1):
    print("Epoch: {0}".format(epoch))
    avg_loss = 0.0


    for t in range(1, n_iter_per_epoch + 1):
        pbar.update(t)
        x_batch, y_batch = next(x_train_generator)
        info_dict = inference.update(feed_dict={
            x_ph: x_batch,
            y_ph: y_batch
        })
        avg_loss += info_dict['loss']

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M
    print("-log p(x) <= {:0.3f}".format(avg_loss))

# actually predict the output
x, y = dataset[:]

pred = y_point.loc.eval({x_ph: x})


print("R2 score is", r2_score(y[:,:10], pred[:,:10]))

from IPython import embed; embed()
