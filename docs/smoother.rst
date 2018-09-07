Biharmonic Smoother
===================

To reduce speckling the velocity fields of coarse resolution SAM, we add a
biharmonic smoother. I got the code from Matt Wyant. The only modification I
made was to set the hyperdiffusion coefficient to be :math:`0.5 \Delta
x^4/\Delta t`. This is equivalent to approximately 2e7 km^4 / s.
