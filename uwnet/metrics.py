import json
from glob import glob

import dask.bag as db
import pandas as pd

import torch
import uwnet.loss
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


def r2_score(truth, pred, mean_dims, dims=None, w=1.0):
    """ R2 score for xarray objects
    """
    if dims is None:
        dims = mean_dims

    mu = truth.mean(mean_dims)
    sum_squares_error = ((truth - pred)**2 * w).sum(dims)
    sum_squares = ((truth - mu)**2 * w).sum(dims)

    return 1 - sum_squares_error / sum_squares


class WeightedMeanSquaredError(Metric):
    """
    Calculates the mean squared error.

    - `update` must receive output of the form `(y_pred, y)`.
    """

    def __init__(self, weights, *args, **kwargs):
        super(WeightedMeanSquaredError, self).__init__(*args, **kwargs)
        self.weights = weights.view(-1, 1, 1) / weights.sum()

    def reset(self):
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        squares = (y - y_pred)**2
        squared_errors = (squares * self.weights).sum()
        self._sum_of_squared_errors += squared_errors.item()
        self._num_examples += y.shape[0] * y.shape[1]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'MeanSquaredError must have at least one example before it can be computed.'
            )
        return self._sum_of_squared_errors / self._num_examples


def read_metrics_files():
    """Read the metrics for all neural networks into a pandas dataframe"""
    jsons = glob('nn/**/*.json', recursive=True)
    json_seq = db.from_sequence(jsons)

    def read_json(path):
        with open(path) as f:
            d = json.load(f)
        d['path'] = path
        return d

    df = json_seq.map(read_json).to_dataframe().compute()

    model_info = df.path.str.extract("(?P<type>.*?)/(?P<model>.*?)/(?P<epoch>.*).json")
    df_with_info = pd.concat([df, model_info], axis=1)
    df_with_info['epoch'] = df_with_info.epoch.astype(int)

    return df_with_info.set_index(['type', 'model', 'epoch', 'path'])
