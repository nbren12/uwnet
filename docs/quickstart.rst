Quickstart
==========

This guide is meant to get a new user to this code base up and running quickly.
This code can be downloaded from github using the following command::

  git clone https://github.com/nbren12/uwnet

Once that is done downloading, you will need to download all the libraries and
packages this code depends upon. To do this, I recommend installing the latest
version of miniconda_. If you have installed that succesfully, the ``conda``
command should be available at the command line.

Installing the dependencies
---------------------------

Now you can change into the ``uwnet`` directory.

This project includes a file ``environment.yaml`` which contains all the details
that miniconda_ needs to configure the environment. The environment can be setup
by running::

  make create_environment

If succesful, this message should appear::

  >>> New Environment created...activate by typing
      source activate uwnet"

From now on, you will need to enter type this command before running any of the
scripts in this project.

Downloading the data
--------------------

The data used for training is a netCDF file that is around 5 GB in size.

.. NOTE::

   The data is available by request at the moment, but I plan to upload it to
   Zenodo soon. The header for the data needs to look like this: :ref:`dataheader`.


Training the neural network
---------------------------

Now that you have downloaded the data and installed the necessary software
dependencies, you should be able to train a neural network by typing in the
following commands::

  python -m uwnet.train with data=<path to dataset>

This script can be trained with many different options, the default options can
be seen using::

  python -m uwnet.train print_config

For instance, to change the learning rate used in the stochastic gradient
descent, you can do::

  python -m uwnet.train with data=<path> lr=.001

By default the script saves the data to a unique directory named like
models/RUNID where RUNID is a unique key in database. This is useful because it
allows keeping track of old experiments, but needs MongoDB to be installed to
work. To avoid this, it is possible to save the data to a directory of your
choice by including ``output_dir=<path to desired output_dir>`` after the
``with``. For example::

  python -m uwnet.train with data=<path> lr=.001 output_dir=<path to desired output_dir>


Training Outputs
----------------

After training this output directory will contain files like these::

  1.pkl
  2.pkl
  q2_1-0.png
  q2_2-0.png
  scatter_fqt_q2_1-0.png
  scatter_fqt_q2_2-0.png


The ``.pkl`` files contain the saved neural network models and the image files
show some diagnostics that helpful for tracking the progress of the training.

.. _miniconda: https://conda.io/miniconda.html

