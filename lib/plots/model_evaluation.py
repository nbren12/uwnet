import holoviews as hv
from itertools import product

from ..hvops import percent, shade_points
from .hexbin import HexBins


def scatter_plot_z(data, true_name, model_names, dim,
                   engine='hexbin'):
    """Create scatter plot of true vs pred for each vertical location
    """

    def scatter(z, xname):
        sub = data.sel(z=z)
        truth = sub.sel(**{dim: true_name}).values.ravel()
        pred = sub.sel(**{dim: xname}).values.ravel()


        if engine == 'hexbin':

            hbin = percent(HexBins((pred, truth), kdims=["prediction", "truth"]))
            return hbin.redim.range(prediction=(0,1))\
                * hv.Curve(((0,1), (0,1))).opts(style=dict(color="black"))
        elif engine == 'points':
            return hv.Points((pred, truth), kdims=["prediction", "truth"])


    z = hv.Dimension('z', values=data.z.values)
    model = hv.Dimension('model', values=model_names)

    hmap = hv.HoloMap({(zz, mm): scatter(zz, mm)
                       for zz, mm in product(z.values, model.values)},
                      kdims=['z', 'model'])

    if engine == 'points':
        return shade_points(hmap)
    else:
        return hmap
