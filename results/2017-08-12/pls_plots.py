import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.externals import joblib

def plot_coef(pkl, weightsfile):
    mod = joblib.load(pkl)
    weights = xr.open_dataarray(weightsfile)

    return plot_coef1(mod, weights)

def plot_coef1(mod, weights):
    """Plot B matrix"""
  

    D = np.diag(weights.values)
    Di = np.diag(1/weights.values)

    def plot_b_pane(D, B, s, ax, z):
        z = np.asarray(z)
        M = xr.DataArray(D@B@Di, {'z_in': z, 'z_out': z}, ('z_in', 'z_out'))
        M.plot(x='z_in', y='z_out', add_colorbar=False, ax=ax, add_labels=False)

        add_label_b(s, ax)

    def add_label_b(s, ax):
        ax.text(.8, .8, s, transform=ax.transAxes)

    def plot_b(B):


        #split output
        Bqt, Bsl = np.split(B, 2, axis=0)
        Bqt_q1, Bqt_q2 = np.split(Bqt, 2, axis=1)
        Bsl_q1, Bsl_q2 = np.split(Bsl, 2, axis=1)

        fig, axs = plt.subplots(2,2, sharex=True, sharey=True)

        plot_b_pane(D, Bqt_q1, "qt vs q1", axs[0,0], weights.z)
        plot_b_pane(D, Bsl_q1, "sl vs q1", axs[0,1], weights.z)
        plot_b_pane(D, Bqt_q2, "qt vs q2", axs[1,0], weights.z)
        plot_b_pane(D, Bsl_q2, "sl vs q2", axs[1,1], weights.z)

        # axs[0,0].set_xlim([0, 16e3])
        # axs[0,0].set_ylim([0, 16e3])

        return axs


    return plot_b(mod.coef_)


def plot_pls_mode(m, x, y, p):
    fig, (a,b) = plt.subplots(1,2, sharey=True)

    x = x.sel(m=m)
    y = y.sel(m=m)

    try:
        a.plot(y.q1, p, label='Q1')
    except AttributeError:
        a.plot(y.q1c, p, label='$Q_{1c}$ ')
    a.plot(y.q2, p, label='Q2')
    b.plot(x.qt, p, label='$q_T$')
    b.plot(x.sl, p, label='$s_l$')

    a.legend()
    b.legend()

    plt.ylim([1000, 10])

    fig.suptitle(f"m = {m+1}")

def pls_plots(file_name, pressure_file):

    x = xr.open_dataset(file_name, group="x_weights")
    y = xr.open_dataset(file_name, group="y_weights")
    p = xr.open_dataset(pressure_file).p


    for i in range(4):
        plot_pls_mode(i, x, y, p)
