from collections import MutableMapping, KeysView
from abc import ABCMeta
from toolz import curry
import attr

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


@attr.s
class TensorDict(MutableMapping, metaclass=ArithmaticMeta):
    """Wrapper which overloads operators for dicts of tensors"""
    data = attr.ib()

    def __getitem__(self, key):
        if isinstance(key, (list, set, KeysView, self.__class__)):
            keys = list(key)
            return TensorDict({key: self.data[key] for key in keys})
        else:
            return self.data[key]

    def copy(self):
        return TensorDict(self.data.copy())

    def apply(self, fun):
        out = self.copy()
        for key in self:
            out[key] = fun(self[key])
        return out
