Training Procedure
==================


Mini-batches
------------

A mini-batch is a collection of time series drawn from random :math:`(x,y)`
locations.

Loss Function
-------------

We use a mass-weighted loss function for 3D variables and a non-mass-weighted one for two dimensional one.

Sequence Prediction
-------------------

The neural network is trained to produce good predictions over multiple time
steps.
