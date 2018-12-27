from stewart_intro.utils import load_data

scalar_features = ['LHF', 'SHF', 'SOLIN']


def get_initial_shape():
    return load_data().QT.shape


def get_target_normalization_with_z(variable, data):
    mean_by_vertical_slice = data[variable].mean(axis=(0, 2, 3)).values
    std = data[variable].std(axis=(0, 2, 3)).values.mean(axis=0)
    return mean_by_vertical_slice, std, data


def get_target_normalization(variable, data):
    if len(data[variable].shape) == 4:
        return get_target_normalization_with_z(variable, data)
    else:
        mean = data[variable].values.mean()
        std = data[variable].values.std()
        return mean, std, data


def normalize_data(filter_down=True):
    """
    Each (x, y) cell has a corresponding 2 * |z| output vector associated
    with it, for a given time point. The 2 variables are SLI and QT.
    """
    data = load_data()
    normalization_dict = {}
    for variable in ['QT', 'SLI', 'SOLIN', 'LHF', 'SHF']:
        mean, sd, data = get_target_normalization(variable, data)
        normalization_dict[variable] = {'mean': mean, 'sd': sd}
        if variable in ['SOLIN', 'LHF', 'SHF']:
            data[variable + '_normalized'] = data[variable].copy()
            data[variable + '_normalized'].values = (
                data[variable].values - mean) / sd
    if filter_down:
        data = data.isel(y=list(range(28, 36)))
    return data, normalization_dict
