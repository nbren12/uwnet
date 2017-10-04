import xarray as xr


def split(snakemake):
    t_split = snakemake.params.t_split
    data = xr.open_dataset(snakemake.input[0],
                           chunks={'time': 100})

    print("Splitting data into training set and testing set")
    print(f"t_split = {t_split}")
    train_slice = slice(None, t_split)
    test_slice = slice(t_split, None)
    data.sel(time=train_slice).to_netcdf(snakemake.output.train)
    data.sel(time=test_slice).to_netcdf(snakemake.output.test)


try:
    snakemake
except NameError:
    pass
else:
    split(snakemake)
