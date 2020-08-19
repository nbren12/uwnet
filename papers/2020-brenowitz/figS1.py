import bin_plots
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

path, output = sys.argv[1:]
ds = xr.open_dataset(path)
fig, axs = plt.subplots(1, 3, figsize=(7, 2), constrained_layout=True)

bin_plots.plot_row(ds, axs=axs)
bin_plots.set_row_titles(axs, ["a) Count\n", "b) Predicted\nP-E (mm/day)", "c) P-E Error\n(mm/day)"])
bin_plots.label_axes(axs[np.newaxis,:])

fig.savefig(output)

