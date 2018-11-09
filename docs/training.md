Training Procedure
==================


Mini-batches
------------

A mini-batch is a collection of time series drawn from random `$ (x,y) $`
locations.

Loss Function
-------------

We use a mass-weighted loss function for 3D variables and a non-mass-weighted one for two dimensional one.

Comparing variables that are two-dimensional with one-dimensional ones requires a little attention. One way to equally weight two dimensional variables in the loss function is to put all variables into energy or power units.

Stabilizing a neural network
----------------------------

There is no guarrantee that a neural network which can predict `$ Q_1 $` and
`$ Q_2 $` will be stable when evaluated over many time steps. In fact it is
almost always not stable, and this lack of stability manifests as an exponential
growth in the underyling fields. This is likely due to the jacobian of the
learned neural network having eigenvalues that lie outside the stability region
of the explicit Euler time stepper we are using to train.

In previous work, we found that training the neural network to produce good
predictions over a sufficient interval in time stabilized the network.
Unfortunately, this approach also produced biased estimate of the apparent
sources. This bias occurs because there is a substantial statisticla difference
between the observed and predicted trajectories starting at a given time point.
This in turn, is likely caused by assuming that the integral of known forcing
`$ g $` in time can be approximate by its starting value `$ g(t^n) $`. While
this estimate itself might not be biased, when integrated into a non-linear time
stepper it can be.

  
A simpler strategy, which also achieves numerical stability of the known time
stepper is to directly penalize the unrealistic growth of the system under a
constant forcing. This removes the issues with bias. Therefore the penalized loss is `$ J(f) = J_1(f) + \lambda J_2(f) $`. The first loss is the error using the trapezoid rule to evaluate the change in the state:
\begin{align}
J_1(f) &= E_n \left| x^{n+1} - \left[ x^{n} + h \frac{g^n + g^{n+1} + f(x^n) + f(x^{n+1})}{2}\right]\right|^2 .
\end{align}
To define the second loss, we introduce an operator which performs an Euler Step with the time-mean forcing for a given location `$i$`:
\begin{align}
\psi_{i,h} x = x + h \left[ \bar{g}_i + f(x) \right].
\end{align}
Here, `$\bar{g}_i$` is the time mean of the forcing for a location `$i$`. Then, the mean-decay penalty term is defined as

\begin{align}
J_2(f; m, h) = E_{i,n} \left| \bar{x_i} - \psi_{i,h}^m x_i^n \right|^2
\end{align}

This expression demands that the learned dynamics approach the equilibrium state
under the time-mean forcings, regardless of the initial conditions. Unlike the
multiple time step loss, this simple equilibrium consistency constraint only
requires that the time-mean of the forcings is estimated well, rather than
demanding that time stepper works well.

In practice, `$ J_1(f) $` is computed for every time and spatial location in a batch, but we only compute `$ J_2(f) $` for a single randomly chosen starting time point `$ n $`. In practice, this is enough to penalize the unstable modes of the neural network time stepper. 
