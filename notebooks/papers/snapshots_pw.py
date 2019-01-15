import matplotlib.pyplot as plt
import xarray as xr

from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap

output = ["snapshots_pw.png"]


def get_data():

    variables = ['PW']
    times = [101, 105, 109]

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
        [
            ng[variables].interp(time=times), nn[variables].interp(time=times),
            micro[variables].interp(time=times)
        ],
        dim=['NG-Aqua', 'NN', 'Micro'])

    return plotme.sel(time=times)


def main():
    plotme = get_data()
    plotme.PW.plot(col='time', row='concat_dim', vmax=55)
    plt.savefig(output[0])


if __name__ == '__main__':
    main()
