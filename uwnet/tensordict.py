import functools
from abc import ABCMeta
from collections import KeysView, MutableMapping

import torch
from toolz import curry, first

mathfuns = [
    '__add__', '__rmul__', '__mul__', '__sub__', '__radd__', '__rmul__',
    '__truediv__', '__floordiv__', '__pow__', '__rpow__'
]

dictfuns = ['__len__', '__iter__', '__delitem__', '__setitem__']


@curry
def valmap_binary_operator(name, self, other):
    out = {}
    for key in self.data:
        func = getattr(self.data[key], name)
        if isinstance(other, self.__class__):
            out[key] = func(other.data[key])
            if self.keys() != other.keys():
                raise ValueError("Both arguments to binary arguments must "
                                 "have the same keys.")
        else:
            out[key] = func(other)
    return TensorDict(out)


@curry
def valmap_unary_operator(name, attrname, self, *args, **kwargs):
    func = getattr(getattr(self, attrname), name)
    return func(*args, **kwargs)


class ArithmaticMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        for fun in mathfuns:
            dct[fun] = valmap_binary_operator(fun)
        for fun in dictfuns:
            dct[fun] = valmap_unary_operator(fun, 'data')
        return super().__new__(cls, name, bases, dct)


class TensorDict(MutableMapping, metaclass=ArithmaticMeta):
    """Wrapper which overloads operators for dicts of tensors"""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if isinstance(key, (list, set, KeysView, self.__class__)):
            keys = list(key)
            return TensorDict({key: self.data[key] for key in keys})
        else:
            return self.data[key]

    def __getattr__(self, key):
        """Get attribute from the values"""
        if 'data' in self.__dict__:
            object = first(self.data.values())
        else:
            raise AttributeError()

        @functools.wraps(getattr(object, key))
        def fun(*args, **kwargs):
            return self.apply(lambda x: getattr(x, key)(*args, **kwargs))

        return fun

    def copy(self):
        return TensorDict(self.data.copy())

    def apply(self, fun):
        out = self.copy()
        for key in self:
            out[key] = fun(self[key])
        return out

    def __repr__(self):
        s = "TensorDict:\n"
        s += "-----------\n"
        for key in self.keys():
            shape = tuple(self.data[key].shape)
            dtype = self.data[key].dtype
            s += f"{key} \t({dtype}): {shape}\n"
        return s

    @functools.wraps(torch.split)
    def split(self, *args, **kwargs):
        outputs = []
        vals = (val.split(*args, **kwargs) for val in self.data.values())
        for vals in zip(*vals):
            d = dict(zip(self.keys(), vals))
            outputs.append(TensorDict(d))
        return outputs
