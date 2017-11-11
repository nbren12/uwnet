Machine learning approaches to convective parametrization
=========================================================

In this project, we will train machine learning models for deep convection
using a near-global aquaplanet cloud permitting simulation as a training
dataset. As with nearly any machine learning task, the first step is to fit a
simple linear model, which in this case is not so simple.


Project Structure
-----------------

Here is an annotated directory tree of this project::

  ├── data
  │   ├── cache
  │   ├── calc
  │   ├── ml
  │   ├── processed 
  │   └── raw       - the raw data (this directory should be immutable)
  ├── docs
  │   └── plots.d   - any .py or .ipynb file in here will be run to make a plot
  ├── lib           - all the source code
  │   ├── models    - machine learning models
  │   ├── plots     
  │   ├── scripts   - snakemake scripts
  │   └── snakemake - some snakemake workflows (might want to remove)
  ├── reports       - generated contents
  ├── results       - mostly has ipython notebooks
  └── snakemake     - contains snakemake stuff





TODO
----

- [x] Fix the results/3.4-ndb-spectra-of-modes.ipynb to compute the actual eigenvalues of the fitted operators. Currently this notebook is incorrect.

Reports
-------

- `3.4-ndb-spectra`_
- `3.6-ndb-pytorch`_
- `3.7-ndb-mca-regression-versus-linear-regression`_

.. _3.4-ndb-spectra: https://storage.googleapis.com/nbren12-data/reports/uw-machine-learning/3.4-ndb-LRF-spectra.html
.. _3.6-ndb-pytorch: https://storage.googleapis.com/nbren12-data/reports/uw-machine-learning/3.6-ndb-pytorch.html
.. _3.7-ndb-mca-regression-versus-linear-regression: https://atmos.washington.edu/~nbren12/reports/3.7-ndb-mca-regression-versus-linear-regression.html
