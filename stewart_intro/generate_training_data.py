from stewart_intro.utils import load_data

scalar_features = ['LHF', 'SHF', 'SOLIN']


def get_initial_shape():
    return load_data().QT.shape


def get_target_normalization(variable):
    data = load_data()
    mean_by_vertical_slice = data[variable].mean(axis=(0, 2, 3)).values
    std_by_vertical_slice = data[variable].mean(axis=(0, 2, 3)).values
    unnormalized = data[variable].values.reshape(
        (
            len(data.time),
            len(data.x),
            len(data.y),
            len(data.z),
        )
    )
    normalized = (
        unnormalized - mean_by_vertical_slice) / std_by_vertical_slice.mean()
    return normalized.reshape(get_initial_shape())


def normalize_data():
    """
    Each (x, y) cell has a corresponding 2 * |z| output vector associated
    with it, for a given time point. The 2 variables are SLI and QT.
    """
    data = load_data()
    data.QT.values = get_target_normalization('QT')
    data.SLI.values = get_target_normalization('SLI')
    return data
