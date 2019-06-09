import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from mpl_toolkits.axes_grid1.colorbar import colorbar



import common
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap

output = ["snapshots_pw.pdf"]


def get_data():

    variables = ['PW']
    time = 105


    # open NN run
    run = runs['debias']
    nn = run.data_2d.rename({'NPNN': 'net_precip'})

    # open NN run
    run = runs['unstable']
    unstable = run.data_2d.rename({'NPNN': 'net_precip'})
    unstable_time = float(unstable.time[-1])

    # open microphysics
    run = runs['micro']
    micro = run.data_2d
    micro['net_precip'] = micro.Prec - lhf_to_evap(micro.LHF)

    # open NGAqua
    ng = open_data('ngaqua_2d')
    ng['net_precip'] = ng.Prec - lhf_to_evap(ng.LHF)
    # make sure the x and y value agree
    ng = ng.assign(x=nn.x, y=nn.y)

    runs_at_time = {
        'NG-Aqua': ng[variables].interp(time=time),
        'NN-Lower': nn[variables].interp(time=time),
        'Base': micro[variables].interp(time=time),
        f'NN-All (t={unstable_time})': unstable[variables].interp(time=unstable_time)
    }
    
    
    return xr.concat(list(runs_at_time.values()), dim=list(runs_at_time.keys()))


def plot_inset_nn_all(ax, pw, xlim=(12e6,16e6), ylim=(4e6,6e6)):
    
    pw = pw.sel(x=slice(*xlim), y=slice(*ylim))
    x, y, z = pw.x / 1e6, pw.y / 1e6, pw.values
    
    c = "w"
    
    styles = {'axes.edgecolor': c,
              'axes.labelcolor': c,
              'xtick.color':c,
              'ytick.color': c
             }
 
    with plt.rc_context(styles):
        axins = zoomed_inset_axes(ax, 2,loc='center left')  # zoom = 6
        im = axins.pcolormesh(x, y, z, rasterized=True)
        # colorbar
        cax = inset_axes(axins,
                     width="100%",  # width = 10% of parent_bbox width
                     height="20%",  # height : 50%
                     loc="lower center",
    #                  bbox_to_anchor=(1.05, 0., 1, 1),
                      bbox_to_anchor=(0.0, -.30, 1., 1),
                     bbox_transform=axins.transAxes,
                     borderpad=0,
                     )

        colorbar(im, cax=cax, orientation='horizontal')

        axins.set_xticklabels('')
        axins.set_yticklabels('')
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec=c)


def plot(plotme):
    w= common.textwidth
    fig = plt.figure(1, figsize=(w, w/1.8))

    grid = AxesGrid(
        fig,
        111,  # similar to subplot(144)
        nrows_ncols=(2, 2),
        aspect=True,
        axes_pad=(.1, .3),
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="3%",
        cbar_pad="2%", )

    count = 0
    abc = 'abcdefghijk'


    cm = plt.cm.viridis
    cm = 'viridis'


    kw = dict(vmin=0, vmax=60, rasterized=True, cmap=cm)

    for j, run in enumerate(plotme.concat_dim):
        
        cax = grid.cbar_axes[count]
        ax = grid[count]

        val = plotme.PW[j]
        x, y, z = val.x / 1e6, val.y / 1e6, val.values

        im = ax.pcolormesh(x, y, z, **kw)
        cax.colorbar(im)

        run = str(run.values)
        ax.set_title(f'{abc[count]}) {run}', loc='left')
        count += 1
        
#         if run.startswith('NN-All'):
#             plot_inset_nn_all(ax, val)

    axs = np.reshape(grid, (2, 2))
    common.label_outer_axes(axs, "x (1000 km)", "y (1000 km)")


if __name__ == '__main__':
    data = get_data()
    plot(data)
    plt.savefig(output[0])
