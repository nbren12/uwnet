import xarray as xr
import pandas as pd
from src.data import open_data, runs
import common
from uwnet.thermo import lhf_to_evap, compute_apparent_source
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

pw_name = 'pw'
netprec_name  = 'net_precip'

def get_ng_and_semiprog():
    # open model
    model = common.get_model('NN-Lower')
    # get data
    ds = open_data('training').sel(time=slice(100,115))

    def integrate_moist(src):
        return (src * ds.layer_mass).sum('z')/1000

    q2 = compute_apparent_source(ds.QT, 86400 * ds.FQT)


    ng = xr.Dataset({
        netprec_name: -integrate_moist(q2),
        pw_name: integrate_moist(ds.QT)
    })

    # semiprognostic
    predicted_srcs = model.predict(ds)
    ds = xr.Dataset({
        netprec_name: -integrate_moist(predicted_srcs['QT']),
        pw_name: integrate_moist(ds.QT)
    })
    
    return ng, ds


def get_data(start_time=100, end_time=120):


    ng = open_data('ngaqua_2d')
    debias = runs['debias'].data_2d
    
    time_slice = slice(start_time, end_time)
    
    #
    ng, semiprog = get_ng_and_semiprog()
    
    # select the data   
    debias = debias.sel(time=time_slice)
    debias = xr.Dataset({
        pw_name: debias.PW,
        netprec_name: debias.NPNN
    })
    
    # merge the data into dataframe
    ng = ng.to_dataframe().assign(run='NG-Aqua')
    debias = debias.to_dataframe().assign(run='NN-Lower')
    semiprog = semiprog.to_dataframe().assign(run='NN-Lower-semi')
    df = pd.concat([ng, debias, semiprog])
    
    return df.reset_index().set_index(['run', 'time', 'x', 'y'])


def hexbin(ax, df, quantiles=[.01, .1, .5, .9, .99],
           y_quantiles=[.01, .1, .5, .9, .99],
           xlim=[32, 55], ylim=[-20, 35]):
#     levels = np.linspace(0, .04, 21)
    df = df.sample(10000)
    im = sns.kdeplot(df.pw, df.net_precip, ax=ax, shade=True, shade_lowest=False)

            
    v = df.pw.mean()
    ax.axvline(v, ls='-', lw=1.5, c='k')
    ax.text(v-.5, 20, f'{v:.1f} mm', horizontalalignment='right', fontsize=9)
        
    v = df.net_precip.mean()
    ax.axhline(v, ls='-', lw=1.5, c='k')
    ax.text(34, v+.5, f'{v:.1f} mm/d', verticalalignment='bottom', fontsize=9)

    ax.set_xlabel('PW (mm)')
    ax.set_ylabel('Net Precip. (mm/d)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return im
    
    
def sel_tropics(run, df):
    return df.loc[run, 110.0:112.0, :, 4.5e6:5.5e6]


def plot(df):
    
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(common.textwidth, 2),
                               sharex=True, sharey=True, constrained_layout=True,
                               )
    a.set_title('a) NG-Aqua', loc='left')
    b.set_title('b) NN-Lower', loc='left')
    c.set_title('c) NN-Lower Semi-prognostic', loc='left')
    hexbin(a, sel_tropics('NG-Aqua', df))
    hexbin(b, sel_tropics('NN-Lower', df))
    hexbin(c, sel_tropics('NN-Lower-semi', df))

    common.despine_axes([a,b, c])
    for ax in [b, c]:
        ax.set_ylabel('')
        ax.spines['left'].set_visible(False)
    fig.set_constrained_layout_pads(wspace=-.05, hspace=0)

    
data = get_data()
plot(data)
plt.savefig("pdf.pdf")