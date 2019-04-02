"""Objects for sampling along xarray objects
"""
import xarray as xr
import numpy as np


class Sampler:
    """Samples along a set of dimensions of an xarray object"""

    def __init__(self,
                 input_variables,
                 output_variables,
                 feature_dims,
                 sample_dims,
                 feature_valid_fn=None):
        self.inputs = input_variables
        self.outputs = output_variables
        self.feature_dims = feature_dims
        self.sample_dims = sample_dims
        self.feature_valid_fn = (feature_valid_fn
                                 if feature_valid_fn else feature_valid)

    def get_valid_features(self, features):
        return [self.feature_valid_fn(var, z) for (var, z) in features]

    def to_input_data_matrix(self, data):
        ds = data.stack(samples=self.sample_dims)
        x = ds[self.inputs].to_stacked_array('feature', self.feature_dims)
        valid_feats = self.get_valid_features(self.feature.values)
        return x[:, valid_feats]

    def to_output_vector(self, data):
        ds = data.stack(sample=self.sample_dims)[self.outputs]
        return ds

    def from_output_vector(self, data, samples):
        return xr.DataArray(
            data, dims='samples', coords={'samples':
                                          samples}).unstack('samples')

    def prepare_data(self, data):
        return self.to_input_data_matrix(data), self.to_output_vector(data)


class SklearnWrapper:
    def __init__(self, sampler, model):
        self.sampler = sampler
        self.model = model

    def predict(self, data):
        x = self.sampler.to_input_data_matrix(data)
        y = self.model.predict(x)
        return self.sampler.from_output_vector(y, x.samples)

    def fit(self, data):
        x, y = self.sampler.prepare_data(data)
        [x_train, y_train], [x_test, y_test] = split_train_test(x, y)
        self.model.fit(x_train, y_train)

        print("Training score", self.model.score(x_train, y_train))
        print("Testing score", self.model.score(x_test, y_test))
        return self


def feature_valid(var, z):
    if var == 'QT':
        if z < 10000:
            return True
        else:
            return False
    if var == 'SLI':
        if z < 10000:
            return True
        else:
            return False
    return True


def split_train_test(x, y):

    nsample = 10000

    valid_feats = [feature_valid(var, z) for (var, z) in x.feature.values]

    inds = np.random.choice(x.shape[0], nsample)
    inds.sort()
    x_train, y_train = x[inds, valid_feats].values, y[inds].values

    inds = np.random.choice(x.shape[0], nsample)
    inds.sort()
    x_test, y_test = x[inds, valid_feats].values, y[inds].values

    return [x_train, y_train], [x_test, y_test]
