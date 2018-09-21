# The System for Atmospheric Modeling

The System for atmospheric modeling (SAM) solves a simplified version of the fluid equations governing the atmosphere known as the anelastic equations. Traditionally, this model is used for high resolution (< 4km) simulations of clouds and convection, but it can also be run in near-global domains at coarse resolution (Bretherton and Khairoutdinov, 2015).

## Prognostic Variables

The main prognostic variables that the System for Atmospheric Modeling uses are:

1. Liquid/ice static energy variable (SLI)
2. Non-precipitating water mixing ratio
3. The wind fields are collocated on an [Arakawa C-grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) 


For more details, see the appendix of Khairoutdinov and Randall (2003).

## Running SAM via docker

We run SAM using docker.

## References

- Bretherton, C. S., & Khairoutdinov, M. F. (2015). Convective self-aggregation feedbacks in near-global cloud-resolving simulations of an aquaplanet. Journal of Advances in Modeling Earth Systems, 7(4), 1765–1787. Retrieved from http://onlinelibrary.wiley.com/doi/10.1002/2015MS000499/full

- Khairoutdinov, M. F., & Randall, D. A. (02/2003). Cloud Resolving Modeling of the ARM Summer 1997 IOP: Model Formulation, Results, Uncertainties, and Sensitivities. Journal of the Atmospheric Sciences, 60(4), 607–625. https://doi.org/10.1175/1520-0469(2003)060<0607:CRMOTA>2.0.CO;2
