import os
import numpy as np
import xarray as xr
from . import ndimage_xarray
from .datasets import tiltwave



def test_ndimage():
    a = tiltwave()
    a.ndimage

    # try gaussian filter
    a.ndimage.gaussian_filter(dict(x=1.0, z=0.0))

    # try one dimensional filter
    a.ndimage.gaussian_filter1d('x', 1.0)
