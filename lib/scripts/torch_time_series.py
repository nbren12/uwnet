#!/usr/bin/env python
"""Fit a torch model for

.. math::

    (x^n+1 - x^n)/dt - g =  f(x^n)

"""
import click
import numpy as np
import torch
from lib.models.torch_models import train_euler_network
from lib.models.multiple_step_objective import train_multistep_objective


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input")
@click.argument("output")
@click.option("-n", default=1)
@click.option("--learning-rate", default=.001)
@click.option("--nsteps", default=1)
@click.option("--nhidden", default=256)
def single(input, output, **kwargs):
    data = np.load(input)
    stepper = train_euler_network(data, **kwargs)
    torch.save(stepper, output)


@cli.command()
@click.argument("input")
@click.argument("output")
@click.option("-n", "--num_epochs", default=1)
@click.option("--num_steps", default=-1)
@click.option("--learning-rate", default=.001)
@click.option("--nsteps", default=1)
@click.option("--nhidden", default=256)
@click.option("--window_size", default=4)
@click.option("--weight_decay", default=.1)
@click.option("--batch_size", default=100)
def multi(input, output, **kwargs):
    data = np.load(input)
    stepper = train_multistep_objective(data, **kwargs)
    torch.save(stepper, output)

if __name__ == '__main__':
    cli()
