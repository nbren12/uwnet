#!/usr/bin/env python
"""Fit a torch model for

.. math::

    (x^n+1 - x^n)/dt - g =  f(x^n)

"""
import click
import numpy as np
import torch
from lib.models.torch_models import train_euler_network




@click.command()
@click.argument("input")
@click.argument("output")
@click.option("-n", default=1)
@click.option("--learning-rate", default=.001)
@click.option("--nsteps", default=1)
@click.option("--nhidden", default=256)
def main(input, output, **kwargs):
    data = np.load(input)
    stepper = train_euler_network(data, **kwargs)
    torch.save(stepper, output)


if __name__ == '__main__':
    main()
