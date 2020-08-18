import torch
import numpy as np
import xarray as xr

from wave.data import open_data, assign_apparent_sources
from wave import thermo

from uwnet.thermo import integrate_q2, integrate_q1, net_precipitation_from_prec_evap


def open_data_with_lts_and_path(data_path):
    ds = xr.open_dataset(data_path).sel(time=slice(120, 140))
    p = ds.p.isel(time=0)

    # make necessary computations
    ds['p'] = p
    ds['lat'] = thermo.ngaqua_y_to_lat(ds.y)
    # compute LTS and Mid Trop moisture
    ds['lts'] = thermo.lower_tropospheric_stability(ds.TABS, p, ds.SST, ds.Ps)
    ds['path'] = thermo.midtropospheric_moisture(ds.QV, p, bottom=850, top=600)

    ds = assign_apparent_sources(ds)
    return ds


def subtropical_data(data_path):
    # load model and datya
    ds = open_data_with_lts_and_path(data_path)
    lat = ds.lat
    # select the sub-tropics
    #     subtropics = (11 < np.abs(lat)) & (np.abs(lat) < 22.5)
    subtropics = (np.abs(lat) < 22.5)  #& (np.abs(lat)>  11.25)
    tropics_lat = (np.abs(lat) < 11.25)

    return ds.isel(y=subtropics)


def compute_nn(model, averages, dims):
    if len(dims) == 1:
        return compute_nn_1d(model, averages, dims[0])
    else:
        return compute_nn_nd(model, averages, dims)


def compute_nn_nd(model, averages, dims):
    new_dim = 'time'  # need to be called "time" unfortunately
    stacked = averages.stack({new_dim: dims}).expand_dims(['x', 'y'])\
        .transpose(new_dim, 'z', 'y', 'x')
    output = model.call_with_xr(stacked)
    return output.unstack(new_dim).squeeze()


def compute_nn_1d(model, averages, dim):
    avgs_expanded = averages.rename({
        dim: 'time'
    }).expand_dims(['x', 'y'], [-1, -2])
    return model.call_with_xr(avgs_expanded).rename({'time': dim}).squeeze()


def groupby_and_compute_nn(tropics, model, key, bins):
    bins_key = key + '_bins'
    averages = (tropics.stack(gridcell=['x', 'y', 'time']).groupby_bins(
        key, bins=bins).mean('gridcell'))

    output = compute_nn(model, averages, batch_dims=[bins_key])
    for key in output:
        NNkey = 'NN' + key
        averages[NNkey] = output[key]

    return averages


def groupby_2d(ds, fun, lts_bins, moisture_bins):
    def _average_over_lts_bins(ds):
        return ds.groupby_bins('lts', lts_bins).apply(fun)

    return (ds.stack(gridcell=['x', 'y', 'time']).groupby_bins(
        'path', bins=moisture_bins).apply(_average_over_lts_bins))


def main(model_path, data_path):
    """Get data averaged over a (LTS, mid trop humidity) space"""

    # bins for moisture and lower tropospheric stability
    moisture_bins = np.arange(15) * 2
    lts_bins = np.r_[7.5:17.5:.5]

    # load data
    tropics = subtropical_data(data_path)
    model = torch.load(model_path)

    # compute averages
    bin_averages = groupby_2d(tropics, lambda x: x.mean('gridcell'), lts_bins,
                              moisture_bins)
    counts = groupby_2d(tropics, lambda x: x.gridcell.count(), lts_bins,
                        moisture_bins)

    # compute nn output and diagnostics within bins
    mass = tropics.layer_mass[0]
    output = compute_nn(model, bin_averages, ['path_bins', 'lts_bins'])
    p_minus_e = -integrate_q2(output.QT, mass).rename('net_precipitation_nn')
    heating = integrate_q1(output.SLI, mass).rename('net_heating_nn')

    # form into one dataset
    return xr.merge([
        tropics.p.rename('p'),
        bin_averages.drop('p'),
        output.rename({
            'QT': 'Q2NN',
            'SLI': 'Q1NN'
        }), p_minus_e, heating,
        counts.rename('count'),
        -integrate_q2(bin_averages.Q2, mass).rename('net_precipitation_src'),
        integrate_q1(bin_averages.Q1, mass).rename('net_heating_src'),
        net_precipitation_from_prec_evap(bin_averages).rename(
            'net_precipitation_from_prec_evap')
    ])


def interval_coordinates_to_midpoint(ds):
    new_coords = {}
    coords = ds.coords

    for dim, coord in coords.items():
        if dim.endswith('bins'):
            new_coords[dim] = np.vectorize(lambda x: x.mid)(coord)
    return ds.assign_coords(**new_coords)


if __name__ == '__main__':
    import sys
    binned = main(sys.argv[1], sys.argv[2])
    output_dataset = binned.pipe(interval_coordinates_to_midpoint).to_netcdf(sys.argv[3])
