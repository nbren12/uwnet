from jacobian import *
from src.data import open_data
import common
import pandas as pd

# bootstrap sample size
n = 20
hatch_threshold = 10

# compute jacobians
training = open_data('training')
training['region'] = common.get_regions(training.y)
tropics = training.isel(y=slice(30,34)).load()
tropics['time_of_day'] = tropics.time % 1
p = tropics.p[0].values

model = common.get_model('NN-All')
samples = list(bootstrap_samples(tropics, n))
jacobians = [get_jacobian(model, sample) for sample in samples]

# make plot
fig, axs = plt.subplots(
    4, 5, figsize=(common.textwidth, common.textwidth-2), sharex=True, sharey=True)
plt.rcParams['hatch.color'] = '0.5'

axs[0,0].invert_yaxis()
axs[0,0].invert_xaxis()
norm = SymLogNorm(1, 2, vmin=-1e5, vmax=1e5)

for ax, jac in zip(axs.flat, jacobians):
    qt_qt = jac['QT']['QT'].detach().numpy()
    im = ax.pcolormesh(p, p, qt_qt, vmin=-2, vmax=2, cmap='RdBu_r', rasterized=True)
    ax.contourf(p, p, qt_qt, levels=[-10, 10], extend='both',
                hatches=['xxxxx', None, 'xxxxx'], colors='none')
    ax.set_aspect(1.0)

common.label_outer_axes(axs, "p (mb)", "p (mb)")
    
plt.colorbar(im, ax=axs.tolist(), fraction=.075)
plt.suptitle(r'$\frac{\partial Q_2}{\partial q_T}$ for 20 Tropical Samples. '
             f'Hatching where |.| > {hatch_threshold:.1f} ');
plt.savefig("bootstrap.pdf")