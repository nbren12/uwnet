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
