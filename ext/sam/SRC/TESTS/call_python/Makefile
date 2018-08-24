CC =gcc


test_f90: test_f90.f90 libplugin.dylib
	gfortran -o $@ -L./ -lplugin $<

test : test.c libplugin.dylib
	$(CC) -o $@ -L./ -lplugin $<

libplugin.dylib : builder.py module.py
	python builder.py

