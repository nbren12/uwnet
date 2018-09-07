Biharmonic Smoother
===================

To reduce speckling the velocity fields of coarse resolution SAM, we add a
biharmonic smoother. I got the code from Matt Wyant. The only modification I
made was to make the hypdediffusivity a namelist constant with a default value
of 1e16 which is 10 times higher than recommended for this resolution in [Jablonowski]_.

.. [Jablonowski] Jablonowski, C. & Williamson, D. L. The Pros and Cons of Diffusion, Filters and Fixers in Atmospheric General Circulation Models. in Numerical Techniques for Global Atmospheric Models (eds. Lauritzen, P., Jablonowski, C., Taylor, M. & Nair, R.) 381–493 (Springer Berlin Heidelberg, 2011).
