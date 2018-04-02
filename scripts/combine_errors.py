import re
import json
import xarray as xr
import pandas as pd

errors = []
dims = []
pattern = re.compile("data/output/model.(.*?)/(.*?)/(.*?)/error.nc")
for f in snakemake.input:
    m = pattern.search(f)
    if m:
        model, seed, epoch = m.groups()
        ds = xr.open_dataset(f)

        arg_file = f"data/output/model.{model}/{seed}/arguments.json"
        args = json.load(open(arg_file))
        # nhidden is a list, so need to just take the first element
        # since all the neural networks I fit are single layer
        args['nhidden'] = args['nhidden'][0]
        args.pop('seed', None)
        ds = ds.assign(**args)

        dims.append((model, seed, int(epoch)))
        errors.append(ds)


names = ['model', 'seed', 'epoch']
dim = pd.MultiIndex.from_tuples(dims, names=names)
dim.name = 'tmp'
ds = xr.concat(errors, dim=dim).unstack('tmp')

ds.to_netcdf(snakemake.output[0])
