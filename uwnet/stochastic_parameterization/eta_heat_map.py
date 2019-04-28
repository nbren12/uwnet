import random
import matplotlib.pyplot as plt
import seaborn as sns


binning_quantiles = [0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1]
show = True


def plot_one_point_in_time(ds):
    time = random.choice(ds.time)
    sns.heatmap(
        ds.sel(time=time).eta.values,
        center=ds.sel(time=time).eta.values.mean(),
        cmap='coolwarm',
        robust=True)
    plt.title(f'Eta Locations for Random Time Point')
    if show:
        plt.show()
    else:
        plt.savefig(
            '/Users/stewart/Desktop/uwnet/eta_locations_for_time_point.png'
        )
    plt.clf()
    plt.close("all")


def plot_full_dataset_heatmaps(ds):
    for eta in range(len(binning_quantiles)):
        occurences_by_location = (ds.eta.values == eta).sum(axis=0)
        sns.heatmap(
            occurences_by_location,
            center=occurences_by_location.mean(),
            cmap='coolwarm',
            robust=True)
        plt.title(f'Number of Training Points for eta = {eta}')
        if show:
            plt.show()
        else:
            plt.savefig(
                '/Users/stewart/Desktop/uwnet/Eta_Heat_Map_Full_Dataset/' +
                f'{eta}.png'
            )
        plt.clf()
        plt.close("all")


def plot_tropis_heatmaps(ds):
    ds_ = ds.isel(y=list(range(28, 36)))
    for eta in range(len(binning_quantiles)):
        occurences_by_location = (ds_.eta.values == eta).sum(axis=0)
        sns.heatmap(
            occurences_by_location,
            center=occurences_by_location.mean(),
            cmap='coolwarm',
            robust=True)
        plt.title(f'Number of Training Points for eta = {eta}')
        if show:
            plt.show()
        else:
            plt.savefig(
                '/Users/stewart/Desktop/uwnet/Eta_Heat_Map_Only_Tropics/' +
                f'{eta}.png'
            )
        plt.clf()
        plt.close("all")