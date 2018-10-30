from stewart_intro.utils import load_data

scalar_features = ['LHF', 'SHF', 'SOLIN']


def get_initial_shape():
    return load_data().QT.shape


def get_target_normalization_with_z(variable, data):
    mean_by_vertical_slice = data[variable].mean(axis=(0, 2, 3)).values
    std = data[variable].std(axis=(0, 2, 3)).values.mean(axis=0)
    array_values = data[variable].values
    for idx, mean in enumerate(mean_by_vertical_slice):
        array_values[:, idx, :, :] = (
            array_values[:, idx, :, :] - mean_by_vertical_slice[idx]) / std
    return array_values


def get_target_normalization(variable, data):
    if len(data[variable].shape) == 4:
        return get_target_normalization_with_z(variable, data)
    else:
        mean = data[variable].values.mean()
        std = data[variable].values.std()
        return (data[variable].values - mean) / std


def normalize_data(filter_down=True):
    """
    Each (x, y) cell has a corresponding 2 * |z| output vector associated
    with it, for a given time point. The 2 variables are SLI and QT.
    """
    data = load_data()
    for variable in ['QT', 'SLI', 'FQT', 'FSLI', 'SOLIN', 'LHF', 'SHF']:
        data[variable].values = get_target_normalization(variable, data)
    if filter_down:
        data = data.isel(y=list(range(30, 35)))
    return data
