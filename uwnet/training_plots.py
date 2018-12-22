import logging

import attr
import matplotlib.pyplot as plt

import xarray as xr

from .columns import single_column_simulation


@attr.s
class TrainingPlotManager(object):
    """Manages the creation of plots during the training process"""

    experiment = attr.ib()
    model = attr.ib()
    dataset = attr.ib()
    single_column_locations = attr.ib()
    interval = attr.ib(default=1)

    def __call__(self, engine):
        return self.plot(engine)

    def plot(self, engine):
        n = engine.state.epoch

        if n % self.interval != 0:
            return

        ex = self.experiment
        imbalance_plot(self.model, self.dataset, engine)
        single_column_plots = [plot_q2(ex), plot_scatter_q2_fqt(ex)]
        for y, x in self.single_column_locations:
            location = self.dataset.isel(
                y=slice(y, y + 1), x=slice(x, x + 1), time=slice(0, 200))

            try:
                output = self.model.call_with_xr(location)
            except ValueError:
                continue

            for plot in single_column_plots:
                plot.save_figure(f'{engine.state.epoch}-{y}', location, output)

            i = engine.state.epoch
            filenames = [
                name + f'{i}-{y}'
                for name in ['qt', 'fqtnn', 'fqtnn-obs', 'pw']
            ]
            # water_budget_plots(self.model, self.dataset, location, filenames)


def water_budget_plots(model, ds, location, filenames):
    nt = min(len(ds.time), 190)
    prognostics = ['QT', 'SLI']
    scm_data = single_column_simulation(
        model, location, interval=(0, nt - 1), prognostics=prognostics)
    merged_pred_data = location.rename({
        'SLI': 'SLIOBS',
        'QT': 'QTOBS',
        'U': 'UOBS',
        'V': 'VOBS',
    }).merge(
        scm_data, join='inner')
    output = model.call_with_xr(merged_pred_data)

    plt.figure()
    scm_data.QT.plot(x='time')
    plt.title("QT")
    plt.savefig(filenames[0])

    plt.figure()
    output.QT.plot(x='time')
    plt.title("FQTNN from NN-prediction")
    output_truth = model.call_with_xr(location)
    plt.savefig(filenames[1])

    plt.figure()
    output_truth.QT.plot(x='time')
    plt.title("FQTNN from Truth")
    plt.xlim([100, 125])
    plt.savefig(filenames[2])

    plt.figure()
    (scm_data.QT * ds.layer_mass / 1000).sum('z').plot()
    (location.QT * ds.layer_mass / 1000).sum('z').plot(label='observed')
    plt.legend()
    plt.savefig(filenames[3])


## Some plotting routines to be called after every epoch
@attr.s
class Plot(object):
    ex = attr.ib()

    def get_filename(self, epoch):
        return self.name.format(epoch)

    def save_figure(self, epoch, location, output):
        fig, ax = plt.subplots()
        self.plot(location, output, ax)
        path = self.get_filename(epoch)
        logging.info(f"Saving to {path}")
        fig.savefig(path)
        self.ex.add_artifact(path)
        plt.close()


class plot_q2(Plot):
    name = 'q2_{}.png'

    def plot(self, location, output, ax):
        return output.QT.plot(x='time', ax=ax)


class plot_scatter_q2_fqt(Plot):
    name = 'scatter_fqt_q2_{}.png'

    def plot(self, location, output, ax):
        x, y, z = [
            x.values.ravel()
            for x in xr.broadcast(location.FQT * 86400, output.QT, output.z)
        ]
        im = ax.scatter(x, y, c=z)
        plt.colorbar(im, ax=ax)


def imbalance_plot(model, dataset, engine):
    from uwnet.thermo import lhf_to_evap
    n = engine.state.epoch
    mass = dataset.layer_mass
    subset = dataset.isel(time=slice(0, None, 20))
    out = model.call_with_xr(subset)
    pme = (subset.Prec - lhf_to_evap(subset.LHF)).mean(['time', 'x'])
    pmenn = -(mass * out.QT).sum('z') / 1000
    pmefqt = ((mass * subset.FQT).sum('z') / 1000 * 86400).mean(['time', 'x'])
    pmenn = pmenn.mean(['time', 'x'])
    plotme = xr.Dataset({
        'truth': pme,
        'nn': pmenn,
        'fqt': pmefqt
    }).to_array(dim='var')

    plt.figure()
    try:
        plotme.plot(hue='var')
    except ValueError:
        df = plotme.squeeze().to_series()
        df.plot(kind='bar')

    plt.savefig(f"{n}-imbalance.png")

    plt.close()
