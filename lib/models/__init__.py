import numpy as np
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..mca import MCARegression
from ..util import weighted_r2_score


def main(get_model, data, output):
    """Fit the model returned by get_model

    Parameters
    ----------
    get_model : callable
        A function which takes the input data as an argument and returns
        the model.
    data : str
        path to data stored in joblib pickle
    output : str
        path to output storage
    """

    data = joblib.load(data)
    x_train, y_train = data['train']
    x_test, y_test = data['test']

    mod = get_model(data)

    print("Fitting", mod)
    mod.fit(x_train, y_train)

    # compute score
    weight_out = data['w'][1]
    y_pred = mod.predict(x_test)
    score = weighted_r2_score(y_test, y_pred, weight_out)
    print("Score", score)

    # compute matrix
    I = np.eye(x_train.shape[1])
    mat = mod.predict(I) - mod.predict(I * 0)
    # mat = np.diag(1/scale_in) @ mat

    output_data = {
        'test_score': score,
        'model': mod,
        'mat': mat,
        'features': {
            'in': x_test.indexes['features'],
            'out': y_test.indexes['features']
        }
    }

    joblib.dump(output_data, output)


def get_linear_model(data):
    mod = make_pipeline(
        VarianceThreshold(.001),
        LinearRegression())

    return mod


def get_mca_mod(data, mod=None):
    """Given data dictionary make scale"""

    scale_in, scale_out = data['scale']
    weight_in, weight_out = data['w']
    mca_scale = (np.sqrt(weight_in) / scale_in,
                 np.sqrt(weight_out) / scale_out)

    if mod is None:
        mod = LinearRegression()

    return MCARegression(
        mod=make_pipeline(StandardScaler(), mod),
        scale=mca_scale,
        n_components=4)


models = {'linear': get_linear_model,
          'mcr': get_mca_mod}
