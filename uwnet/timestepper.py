"""Time steppers"""
import attr
from toolz import merge


@attr.s
class Batch(object):
    """A object for getting appropriate fields from a batch of data"""
    data = attr.ib()
    prognostics = attr.ib()

    @property
    def forcings(self):
        return set(self.data.keys()) - set(self.prognostics)

    def get_forcings_at_time(self, t):
        return select_keys_time(self.data, self.forcings, t)

    def get_known_forcings_at_time(self, t):
        out = {}
        for key in self.prognostics:
            original_key = 'F' + key
            if original_key in self.data:
                out[key] = self.data[original_key][t]
        return out

    def get_prognostics_at_time(self, t):
        return select_keys_time(self.data, self.prognostics, t)


def predict_one_step(prognostics, apparent_source, forcing, time_step):
    prediction = {}
    for key in prognostics:
        total_source = (apparent_source[key] / 86400 + forcing[key])
        prediction[key] = prognostics[key] + time_step * total_source
    return prediction


def predict_multiple_steps(model, batch, initial_time, prediction_length,
                           time_step):
    """Yield Euler step predictions with a neural network"""
    prognostics = batch.get_prognostics_at_time(initial_time)
    for t in range(initial_time, initial_time + prediction_length):
        forcings = batch.get_forcings_at_time(t)
        known_forcing = batch.get_known_forcings_at_time(t)
        inputs = merge(forcings, prognostics)
        apparent_sources = model(inputs)
        prognostics = predict_one_step(prognostics, apparent_sources,
                                       known_forcing, time_step)
        yield t + 1, prognostics


def select_keys_time(x, keys, t):
    return {key: x[key][t] for key in keys}
