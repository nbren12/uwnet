import random
import xarray as xr
import pickle

project_dir = '/Users/stewart/projects/uwnet/'
data_dir = project_dir + '../data/'
dataset_filename = \
    '2018-09-18-NG_5120x2560x34_4km_10s_QOBS_EQX-SAM_Processed.nc'


def load_data():
    file_location = data_dir + dataset_filename
    return xr.open_dataset(file_location)


def pickle_model(obj, filename):
    path = project_dir + 'stewart_intro/models/' + filename + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_model(filename):
    path = project_dir + 'stewart_intro/models/' + filename + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_subsampled_dataset(proportion_of_initial_data=0.1):
    data = load_data()
    proportion_by_feature = proportion_of_initial_data ** .5
    n_x = round(len(data.x) * proportion_by_feature)
    n_y = round(len(data.y) * proportion_by_feature)
    x_to_keep = random.sample(list(data.x.values), n_x)
    y_to_keep = random.sample(list(data.y.values), n_y)
    data.sel(x=x_to_keep, y=y_to_keep)
    return data
