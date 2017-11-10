"""Holoviews operations
"""
import param
import holoviews as hv
from holoviews.operation import Operation
from holoviews.operation.datashader import (aggregate, datashade, dynspread,
                                            shade)


class percent(Operation):
    """
    Subtracts two curves from one another.
    """

    label = param.String(
        default='Percent',
        doc="""
        Normalizes data between 0 and 1.
        """)

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def _process(self, element, key=None):

        # Get x-values and y-values of curves
        x = element.dimension_values(0)
        y = element.dimension_values(1)
        M, m = y.max(), y.min()

        x = (x - m) / (M - m)
        y = (y - m) / (M - m)

        # Return new Element with subtracted y-values
        # and new label
        #         print("Scatter")
        return element.clone((x, y))


def shade_points(scat):
    return datashade(percent(scat), cmap="blue")\
        .redim.range(true=(0,1), lm=(0,1)) \
        .redim.unit(lm="%", true="%")\
        * hv.Curve(((0,1),(0,1))).opts(style=dict(color='black'))


def ndoverlay_to_contours(ndoverlay):
    """Turn an NDOverlay into a contour

    Notes
    -----
    Provided by Phillip Rudiger on the gitter channel

    Examples
    --------
    >>>     def ndoverlay_to_contours(ndoverlay):
    ...         dims = ndoverlay.dimensions(label='name')
    ...         return hv.Contours([dict(v.columns(), **dict(zip(dims, k))) for k, v in ndoverlay.data.items()],
    ...                         kdims=ndoverlay.last.dimensions()[:2], vdims=ndoverlay.kdims)
    ...
    ...     curves = hv.HoloMap({(chr(n+65), n): hv.Curve(np.random.rand(10)) for n in range(10)}, kdims=['variable', 'n'])
    ...     curves.overlay('n').grid('variable').map(ndoverlay_to_contours, hv.NdOverlay)
    ...
    """
    dims = ndoverlay.dimensions(label='name')
    return hv.Contours(
        [
            dict(v.columns(), **dict(zip(dims, k)))
            for k, v in ndoverlay.data.items()
        ],
        kdims=ndoverlay.last.dimensions()[:2],
        vdims=ndoverlay.kdims)
