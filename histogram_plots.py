#!/usr/bin/env python
# coding: utf-8

# In `heating_dependence_on_lts_moisture`, I produce plots of varying Q1 and Q2 for the mean within mid-tropospheric moisture (Q) and lower-tropospheric stability (LTS) bins **separately**. Since Q and LTS covary, it is important to study them in a two dimensional phase space. This notebook makes similar plots, but conditioned on a given value of "LTS".

# # Version information

# In[1]:

# # Functions

# In[2]:

# Adjustments to path
import os
import shutil
import xarray as xr
import matplotlib.pyplot as plt
from plots.plot_binned import plot_line_by_key_altair
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Make plots of bin averaged data')

    parser.add_argument('binned_data')
    parser.add_argument('prefix')
    parser.add_argument('--lts-margin', default=3, type=int)
    parser.add_argument('--path-margin', default=10, type=int)

    return parser.parse_args()


args = parse_arguments()

path = args.binned_data
LTS_MARGIN = args.lts_margin
PATH_MARGIN = args.path_margin
pre = args.prefix

binned = xr.open_dataset(path)

os.makedirs('figs', exist_ok=True)
os.chdir('figs')

ds = binned.isel(lts_bins=LTS_MARGIN)
chart = plot_line_by_key_altair(
    ds,
    'path',
    c_title='Q (mm)',
    title_fn=lambda x: f'LTS bin: {x.lts_bins.item(): 03.1f} (K)',
    cmap='viridis',
    c_sort="ascending")
chart.save(pre + "vary_q.svg")

moist_margin = binned.isel(path_bins=PATH_MARGIN)
chart = plot_line_by_key_altair(
    moist_margin,
    'lts',
    c_title='LTS (K)',
    title_fn=lambda x:
    f'Mid tropospheric humidity bin: {x.path_bins.item(): 03.1f} (mm)',
    cmap='viridis')
chart.save(pre + "vary_lts.svg")
