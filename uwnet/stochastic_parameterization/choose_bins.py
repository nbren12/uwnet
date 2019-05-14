from uwnet.stochastic_parameterization.utils import (
    get_dataset,
)
import numpy as np
dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
base_model_location = dir_ + 'full_model/1.pkl'
ds_location = dir_ + 'training.nc'
data = get_dataset(
    t_start=400,
    t_stop=500,
    base_model_location=base_model_location,
    ds_location=ds_location,
    binning_quantiles=(1,)
).column_integrated_qt_residuals.values.ravel()


def get_bin_membership(binning_quantiles):
    bin_membership = []
    prev_quantile = 0
    for quantile in binning_quantiles:
        bin_membership.append(data[
            (data >= np.quantile(data, prev_quantile)) &
            (data < np.quantile(data, quantile))
        ])
        prev_quantile = quantile
    return bin_membership


def loss(binning_quantiles):
    return sum(
        len(bin_) * bin_.var() for
        bin_ in get_bin_membership(binning_quantiles)
    )


if __name__ == '__main__':
    print(loss((.3, .6, 1)))
