from my_plugin import ffi
import numpy as np

@ffi.def_extern()
def hello_world():
    print("Hello World!")

@ffi.def_extern()
def add_one(c, n):
    buf = ffi.buffer(c, n * 8)
    array = np.frombuffer(buf, count=n, dtype='float64')
    array[:] = np.arange(n) * 1.0
