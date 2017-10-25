import xarray as xr


def wrap_xarray_calculation(f):
    def fun(*args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, str):
                new_args.append(xopena(a))
            else:
                new_args.append(a)

        return f(*new_args, **kwargs)

    return fun


def xopen(name, nt=20):
    return xr.open_dataset(name, chunks=dict(time=nt))\
             .apply(lambda x: x.squeeze())


def xopena(name, nt=20):
    # find variable which isn't p
    f = xr.open_dataset(name, chunks={'time': nt})
    varname = [v for v in f.data_vars if v!='p'][0]
    return f[varname]
