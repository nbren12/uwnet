

betts_miller.o: betts_miller.f90 sat_vapor_pres.o
sat_vapor_pres.o : sat_vapor_pres.f90 constants.o

%.o : %.f90
	gfortran -c $<

clean:
	rm -f *.o
	rm -f *.mod
