"""Time steppers"""
import attr
from toolz import merge, first
from .tensordict import TensorDict


@attr.s
class Batch(object):
    """A object for getting appropriate fields from a batch of data"""
    data = attr.ib(converter=TensorDict)
    prognostics = attr.ib()

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
        item = first(self.data.values())
        return item.shape[1]

    @property
    def size(self):
        item = first(self.data.values())
        return item.shape[0]

    def get_time_mean(self, key):
        return self.data[key].mean(-4)

def predict_one_step(prognostics, apparent_source, forcing, time_step):
    prediction = {}
    for key in prognostics:
        total_source = (apparent_source[key] / 86400 + forcing[key])
        prediction[key] = prognostics[key] + time_step * total_source
    return prediction


def predict_multiple_steps(model, batch: Batch, initial_time,
                           prediction_length, time_step):
    """Yield Euler step predictions with a neural network"""
    state = batch.get_prognostics_at_time(initial_time)
    for t in range(initial_time, initial_time + prediction_length):
        inputs = batch.get_model_inputs(t, state)
        apparent_sources = model(inputs)
        known_forcing = batch.get_known_forcings_at_time(t)
        state = predict_one_step(state, apparent_sources,
                                 known_forcing, time_step)
        yield t + 1, state


def select_keys_time(x, keys, t):
    return {key: x[key][:, t] for key in keys}
