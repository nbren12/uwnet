import numpy as np

import param
import holoviews as hv
from holoviews import Dimension
from holoviews.plotting.mpl import ColorbarPlot, ElementPlot


class HexBins(hv.Points):

    kdims = param.List([Dimension('x'), Dimension('y')], bounds=(2, 2))

    group = param.String(default='HexBins')


class HexBinPlot(ColorbarPlot):

    agg_func = param.Callable(default=np.sum)

    gridsize = param.Integer(default=50)

    style_opts = ['edgecolors', 'alpha',
                  'linewidths', 'marginals', 'cmap',
                  'norm']

    _plot_methods = dict(single='hexbin')

    def get_data(self, element, ranges, style):
        if not element.vdims:
            element = element.add_dimension('z', 0, np.ones(len(element)),
                                            True)
        xs, ys = (element.dimension_values(i) for i in range(2))
        args = (ys, xs) if self.invert_axes else (xs, ys)
        args += (element.dimension_values(2), )

        cdim = element.vdims[0]
        self._norm_kwargs(element, ranges, style, cdim)
        style['reduce_C_function'] = self.agg_func
        style['vmin'], style['vmax'] = cdim.range
        style['xscale'] = 'log' if self.logx else 'linear'
        style['yscale'] = 'log' if self.logy else 'linear'
        style['gridsize'] = self.gridsize
        return args, style, {}

hv.Store.register({HexBins: HexBinPlot}, 'matplotlib')
