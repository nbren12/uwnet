import os
import xarray as xr
import matplotlib.pyplot as plt
from lib.plots import plot_soln
import jinja2


def savefig(name):
    plt.savefig(os.path.join(plots_dir, name))


def plot_columns(cols):
    d = xr.open_dataset(cols)
    d = d.isel(x=8, y=8)
    plot_soln(d)


i = snakemake.input
o = snakemake.output

root = os.path.dirname(o[0])
plots_dir = os.path.join(root, "_plots")


if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

figures = {}

plot_columns(i[0])
savefig("cols.png")
figures['Single Column'] = 'cols.png'


template ="""
{% for fig, path in figures.items() %}
<h2>{{fig}}</h2>
<img src="_plots/{{path}}" width="800px"></img>
{% endfor %}
"""

s = jinja2.Template(template).render(figures=figures)
with open(snakemake.output[0], "w") as f:
    f.write(s)
