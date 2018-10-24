Training Procedure
==================


Mini-batches
------------

A mini-batch is a collection of time series drawn from random :math:`(x,y)`
locations.

Loss Function
-------------

We use a mass-weighted loss function for 3D variables and a non-mass-weighted one for two dimensional one.

Comparing variables that are two-dimensional with one-dimensional ones requires a little attention. One way to equally weight two dimensional variables in the loss function is to put all variables into energy or power units.

Sequence Prediction
-------------------

The neural network is trained to produce good predictions over multiple time
steps.

Script
------

.. automodule:: uwnet.train
