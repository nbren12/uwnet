import attr
import click
import torch

import xarray as xr
from uwnet.timestepper import Batch, predict_multiple_steps


def _convert_dataset_to_dict(dataset):
    return {
        key: dataset[key]
        for key in dataset.data_vars if 'time' in dataset[key].dims
    }


class XarrayBatch(Batch):
    """An Xarray-aware version of batch"""

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        data = _convert_dataset_to_dict(dataset)
        super(XarrayBatch, self).__init__(data, **kwargs)

    @staticmethod
    def select_time(data, t):
        return data.apply(lambda x: x.isel(time=t))

    def get_model_inputs(self, t, state):
        inputs = super(XarrayBatch, self).get_model_inputs(t, state)
        for key in inputs:
            try:
                inputs[key] = inputs[key].drop('time')
            except ValueError:
                pass
        return xr.Dataset(inputs)

    def get_prognostics_at_time(self, t):
        return self._dataset[self.prognostics].isel(time=t)


def _get_time_step(ds):
    return float(ds.time.diff('time')[0] * 86400)


def single_column_simulation(model,
                             dataset,
                             interval=None,
                             prognostics=(),
                             time_step=None):
    """Run a single column model simulation with a model for the source terms

    Parameters
    ----------
    model
        pytorch model for producing the apparent sources
    dataset : xr.Dataset
        input dataset in the same format as the training data
    interval : tuple
        (start_time, end_time) interval
    """
    if not time_step:
        time_step = _get_time_step(dataset)

    if not interval:
        start, end = 0, len(dataset.time) - 1
    else:
        start, end = interval

    batch = XarrayBatch(dataset, prognostics=prognostics)
    pred_generator = predict_multiple_steps(
        model.call_with_xr,
        batch,
        initial_time=start,
        prediction_length=end - start,
        time_step=time_step)
    datasets = []
    for k, state, diag in pred_generator:

        for key in diag:
            if key in state:
                diag = diag.rename({key: 'F' + key + 'NN'})

        datasets.append(xr.Dataset(state).assign_coords(time=dataset.time[k]).merge(diag))
    output_time_series = xr.concat(datasets, dim='time')
    return output_time_series


def compute_apparent_sources(model, ds):
    sources = model.call_with_xr(ds)
    rename_dict = {}
    for key in sources.data_vars:
        rename_dict[key] = 'F' + key + 'NN'
    return sources.rename(rename_dict)


def remove_nonphysical_dims(data):
    """Remove all dimensions other than x y z or time"""
    true_dims = ['x', 'y', 'z', 'time']
    for dim in data.dims:
        if dim not in true_dims:
            data = data.isel(**{dim: 0})
    return data


@click.command()
@click.argument('model')
@click.argument('data')
@click.argument('output_path')
@click.option('-b', '--begin', type=int)
@click.option('-e', '--end', type=int)
def main(model, data, output_path, begin, end):
    model = torch.load(model)
    data = xr.open_dataset(data)

    if begin is None:
        begin = 0
    if end is None:
        end = len(data.time)

    data = data.isel(time=slice(begin, end))
    data = remove_nonphysical_dims(data)
    output = single_column_simulation(model, data)
    sources = compute_apparent_sources(model, data)
    output.merge(sources).to_netcdf(output_path)


if __name__ == '__main__':
    main()
