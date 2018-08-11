import xarray as xr
import zarr


def rechunk(dest, var, chunks):
    dims = var.attrs['_ARRAY_DIMENSIONS']
    chunks = [chunks[key] for key in dims]
    arr = dest.zeros(var.name, shape=var.shape, chunks=chunks)
    arr[:] = var[:]

    for key in var.attrs:
        arr.attrs[key] = var.attrs[key]

    return arr


def rename_dims(var):
    dims = var.attrs['_ARRAY_DIMENSIONS']
    out = []
    for dim in dims:
        if dim in ('xs', 'xc'):
            dim = 'x'
        elif dim in ('ys', 'yc'):
            dim = 'y'
        out.append(dim)

    var.attrs['_ARRAY_DIMENSIONS'] = out


infile = "./training_data.zarr"
out = "rechunk.zarr"
n = 1
chunks = {
    'x': n,
    'y': n,
    'time': 640,
    'xc': n,
    'xs': n,
    'ys': n,
    'yc': n,
    'z': 34
}

src = zarr.open_group(infile)
dest = zarr.open_group(out, mode='w')

for key in src.attrs:
    dest.attrs[key] = src.attrs[key]

for key in src:
    print("Processing", key)
    rechunk(dest, src[key], chunks)

for key in dest:
    rename_dims(dest[key])
