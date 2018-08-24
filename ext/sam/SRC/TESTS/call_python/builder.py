# file plugin_build.py
import cffi
ffibuilder = cffi.FFI()


header = open("plugin.h").read()
module = open("module.py").read()

ffibuilder.embedding_api(header)

ffibuilder.set_source("my_plugin", r'''
    #include "plugin.h"
''')

ffibuilder.embedding_init_code(module)

ffibuilder.compile(target="libplugin.dylib", verbose=True)
