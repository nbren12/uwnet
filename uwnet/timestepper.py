"""Time steppers"""
import attr
from toolz import merge, first
from . import tensordict
from .tensordict import TensorDict


@attr.s
class Batch(object):
    """A object for getting appropriate fields from a batch of data

    Attributes
    ----------
    data
    prognostics
    time_dim
    batch_dim
    """
    data = attr.ib(converter=TensorDict)
    prognostics = attr.ib()
    time_dim = 1
    batch_dim = 0

    @property
    def forcings(self):
        return set(self.data.keys()) - set(self.prognostics)

    @staticmethod
    def select_time(data, t):
        return data.apply(lambda x: x[:, t])

    def get_known_forcings(self):
        out = {}
        for key in self.prognostics:
            original_key = 'F' + key
            if original_key in self.data:
                out[key] = self.data[original_key]
        return TensorDict(out)

    def get_prognostics(self):
        return self.data[self.prognostics]

    def get_forcings_at_time(self, t):
        return self.select_time(self.data[self.forcings], t)

    def get_known_forcings_at_time(self, t):
        return self.select_time(self.get_known_forcings(), t)

    def get_prognostics_at_time(self, t):
        return self.select_time(self.data[self.prognostics], t)

    def get_model_inputs(self, t, prognostics=None):
        forcings = self.get_forcings_at_time(t)
        if prognostics is None:
            return TensorDict(merge(forcings, self.get_prognostics_at_time(t)))
        else:
            return TensorDict(merge(forcings, prognostics))

    @property
    def num_time(self):
        return self.data.size(self.time_dim)

    @property
    def size(self):
        """The batch size"""
        return self.data.size(self.batch_dim)

    def get_time_mean(self, key):
        return self.data[key].mean(self.time_dim)

    def data_for_lag(self, lag):
        return tensordict.lag(self.data[self.prognostics], lag, self.time_dim)


class TimeStepper:

    def __init__(self, source_function, time_step):
        self.source_function = source_function
        self.time_step = time_step

    def __call__(self, batch, initial_time=0, prediction_length=None):

        if prediction_length is None:
            prediction_length = batch.num_time - initial_time

        steps = predict_multiple_steps(
            self.source_function, batch, initial_time, prediction_length,
            self.time_step)
        state = [state for t, state, diags in steps]
        return tensordict.stack(state, dim=batch.time_dim)


def predict_multiple_steps(model, batch: Batch, initial_time,
                           prediction_length, time_step):
    """Yield Euler step predictions with a neural network"""
    state = batch.get_prognostics_at_time(initial_time)
    # yield initial_time, state, {}
    for t in range(initial_time, initial_time + prediction_length):
        inputs = batch.get_model_inputs(t, state)
        apparent_sources = model(inputs)
        known_forcing = batch.get_known_forcings_at_time(t)
        state = state + (apparent_sources/86400 + known_forcing) * time_step
        yield t + 1, state, apparent_sources


def select_keys_time(x, keys, t):
    return {key: x[key][:, t] for key in keys}
