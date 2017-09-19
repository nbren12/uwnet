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
    return xr.open_dataarray(name, chunks={'time': nt})
