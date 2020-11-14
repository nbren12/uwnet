"""A module containing useful patches to xarray
"""
import functools
import inspect
from functools import reduce
from operator import mul

import dask.array as da
import numpy as np
import scipy.ndimage
import xarray as xr
from scipy.interpolate import interp1d


# ndimage wrapper
class MetaNdImage(type):
    def __new__(cls, name, parents, dct):

        # for each function in scipy.ndimage wrap and add to class
        for func_name, func in inspect.getmembers(scipy.ndimage, inspect.isfunction):
            if func_name[-2:] == '1d':
                dct[func_name] = MetaNdImage.wrapper1d(func)
            else:
                dct[func_name] = MetaNdImage.wrappernd(func)
            # setattr(xr.DataArray, 'ndimage_' + func_name, ndimage_wrapper(func))
        return super(MetaNdImage, cls).__new__(cls, name, parents, dct)


    def wrappernd(func):
        """Wrap a subset of scipy.ndimage functions for easy use with xarray"""

        @functools.wraps(func)
        def f(self, axes_kwargs, *args, dims=[], **kwargs):

            x = self._obj
            # named axes args to list
            axes_args = [axes_kwargs[k] for k in x.dims]
            y = x.copy()

            axes_args.extend(args)
            y.values = func(x, axes_args, **kwargs)
            y.attrs['edits'] = repr(func.__code__)

            return y

        return f


    def wrapper1d(func):
        """Wrapper for 1D functions
        """

        @functools.wraps(func)
        def f(self, dim, *args, **kwargs):

            x = self._obj
            # named axes args to list
            y = x.copy()
            y.values = func(x, *args, axis=x.get_axis_num(dim), **kwargs)
            y.attrs['edits'] = repr(func.__code__)

            return y

        return f


@xr.register_dataarray_accessor('ndimage')
class NdImageAccesor(metaclass=MetaNdImage):
    def __init__(self, obj):
        self._obj = obj
