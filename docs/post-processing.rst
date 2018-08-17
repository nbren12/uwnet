Post Processing NGAqua
======================

There are several steps needed before the high resolution NGAqua data is usable for training neural networks

Coarse-Graining
---------------

First, the data must be coarse-grained onto 160 km grid boxes in a manner that
respects the staggering of the original variables. This means that variables
that are collocated in the center of grid cells, such as temperature, humidity,
and vertical velocity should be averaged over the entire coarse graining region.
On the other hand, the horizontal velocities are collocated on the cell
interface. Therefore, the coarse-grained horizontal velocities should be
obtained by averaging along the cell edges.

This pipeline is performed on the olympus machine at UW using this `workflow
<https://github.com/nbren12/gnl/tree/master/workflows/sam/coarsegrain>`_.

Post-processing
---------------

The NGAqua data has some peculariaties. For example, the domain mean of the
vertical velocity does not vanish as it should, which indicates that NGAqua does
not conserve mass in the anelastic sense. However, we can use SAMs pressure
solver to enforce mass conservation by providing the NGAqua data as an initial
condition to SAM and saving the velocity field after calling the pressure
solver.
