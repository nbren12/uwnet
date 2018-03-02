from .single_column_initial_value import plot_soln
from . import climatology
from .common_data import load_data


def hide_xlabels(x):
    x.xaxis.set_ticklabels([])
    x.set_xlabel('')
