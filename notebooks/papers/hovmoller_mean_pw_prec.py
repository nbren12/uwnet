import xarray as xr

from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap
import matplotlib.pyplot as plt

outputs = ["hov_mean_prec.png", "hov_mean_pw.png"]


def get_data():

    variables = ['PW', 'net_precip']
    times = slice(100, 110)

    # open NN run
    run = runs['debias']
    nn = run.data_2d.rename({'NPNN': 'net_precip'})

    # open microphysics
    run = runs['micro']
    micro = run.data_2d
    micro['net_precip'] = micro.Prec - lhf_to_evap(micro.LHF)

    # open NGAqua
    ng = open_data('ngaqua_2d')
    ng['net_precip'] = ng.Prec - lhf_to_evap(ng.LHF)
    # make sure the x and y value agree
    ng = ng.assign(x=nn.x, y=nn.y)

    plotme = xr.concat(
        [ng[variables].interp(time=nn.time), nn[variables], micro[variables]],
        dim=['NG-Aqua', 'NN', 'Micro'])

    return plotme.sel(time=times).mean('x')


def main():
    plotme = get_data()
    plotme.net_precip.plot(col='concat_dim', x='time')
    plt.savefig(outputs[0])
    plotme.PW.plot(col='concat_dim', x='time')
    plt.savefig(outputs[1])


if __name__ == '__main__':
    main()
