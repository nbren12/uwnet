import os
import jinja2
from lib.plots.single_column_initial_value import main

i = snakemake.input
o = snakemake.output

root = os.path.dirname(o[0])
plots_dir = os.path.join(root, "_plots")


if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

figures = {}

figures.update(main(i.x, i.f, i.mod, plots_dir))



template ="""
{% for fig, path in figures.items() %}
<h2>{{fig}}</h2>
<img src="_plots/{{path}}" width="800px"></img>
{% endfor %}
"""

s = jinja2.Template(template).render(figures=figures)
with open(snakemake.output[0], "w") as f:
    f.write(s)
