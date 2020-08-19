import json
from json import JSONEncoder
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def dump(arg, f):
    """Serialize a linearized response function dictionary"""
    return json.dump(arg, f, cls=NumpyArrayEncoder)


def lists_to_ndarray(d):
    if isinstance(d, dict):
        return {key: lists_to_ndarray(d[key]) for key in d}
    elif isinstance(d, list):
        return np.array(d)
    else:
        return d

def load(f):
    """Deserialize a linearized response function from file"""
    loaded = json.load(f)
    return lists_to_ndarray(loaded)


