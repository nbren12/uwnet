import holoviews as hv
from holoviews.operation.datashader import (aggregate, datashade, dynspread,
                                            shade)

from ..hvops import percent


def scatter_plot_z(data, xname, yname, dim):
    """Create scatter plot of true vs pred for each vertical location
    """
    df = data.to_dataset(dim=dim).to_dataframe()
    tab = hv.Table(df.reset_index())
    scat = (tab.to.scatter(kdims=[xname], vdims=[yname], groupby="z"))


    return datashade(percent(scat), cmap="blue")\
    .redim.range(true=(0,1), lm=(0,1)) \
    .redim.unit(lm="%", true="%")\
    * hv.Curve(((0,1),(0,1))).opts(style=dict(color='black'))
