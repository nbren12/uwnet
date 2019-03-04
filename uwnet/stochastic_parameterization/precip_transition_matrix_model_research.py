import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from uwnet.stochastic_parameterization.utils import get_dataset


ds = get_dataset(binning_method='precip')
# ds = ds.isel(y=range(1, len(ds.y) - 1))
etas = list(range(ds.eta.values.max() + 1))
poly_degree = 3
model = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000)
# model = GradientBoostingClassifier()


def analyze_linear_regression_model(x_data, y_data, fit_intercept=True):
    if fit_intercept:
        x_data = sm.add_constant(x_data)
    model = sm.OLS(y_data, x_data)
    results = model.fit()
    print(results.summary())


def get_col_names():
    eta_col_names = [f'eta_{eta}' for eta in etas]
    sst_col_names = [
        f'sst_{sst_degree}' for sst_degree in range(1, poly_degree + 1)
    ]
    return eta_col_names + sst_col_names


def normalize_array(array):
    return (array - array.mean()) / array.std()


def get_sst_data(times):
    sst = ds.isel(time=times).SST.values.ravel()
    sst = np.array([
        normalize_array(sst) ** degree for degree in range(1, poly_degree + 1)
    ])
    return np.stack(sst).T


def format_training_data():
    start_times = np.array(range(len(ds.time) - 1))
    stop_times = start_times + 1
    start = ds.isel(time=start_times).eta.values.ravel()
    y_data = ds.isel(time=stop_times).eta.values.ravel()
    x_data = np.zeros((len(start), len(etas)))
    x_data[np.arange(len(y_data)), start] = 1
    if poly_degree:
        sst_data = get_sst_data(start_times)
        x_data = np.hstack((x_data, sst_data))
    return pd.DataFrame(x_data, columns=get_col_names()), y_data


def get_transition_matrix(dataset):
    start_times = np.array(range(len(dataset.time) - 1))
    stop_times = start_times + 1
    eta_start = dataset.isel(time=start_times).eta.values.ravel()
    eta_stop = dataset.isel(time=stop_times).eta.values.ravel()
    transition_matrix = np.zeros((len(etas), len(etas)))
    for eta_row in etas:
        denominator = (eta_start == eta_row).sum()
        for eta_col in etas:
            numerator = (
                (eta_start == eta_row) &
                (eta_stop == eta_col)
            ).sum()
            transition_matrix[eta_row, eta_col] = numerator / denominator
    return transition_matrix


def get_transition_matrix_from_model(model, additional_input):
    transition_matrix = []
    for eta in etas:
        input_ = np.zeros(len(etas))
        input_[eta] = 1
        input_ = np.concatenate([input_, additional_input]).reshape(1, -1)
        transition_matrix.append(model.predict_proba(input_)[0])
    return np.array(transition_matrix)


def get_sst_specific_transition_matrices():
    for y_idx, sst in enumerate(normalize_array(ds.SST.values[0, :, 0])):
        ds_for_sst = ds.isel(y=y_idx)
        print(y_idx)
        print(get_transition_matrix(ds_for_sst).round(2))
        print('\n\n')


def get_modeled_sst_specific_transition_matrices(model):
    for y_idx, sst in enumerate(normalize_array(ds.SST.values[0, :, 0])):
        if poly_degree:
            additional_input = [
                sst ** degree for degree in range(1, poly_degree + 1)
            ]
        else:
            additional_input = []
        print(y_idx)
        print(get_transition_matrix_from_model(
            model, additional_input).round(2))
        print('\n\n')


def get_transition_matrix_mse(model):
    errors = []
    for y_idx, sst in enumerate(normalize_array(ds.SST.values[0, :, 0])):
        ds_for_sst = ds.isel(y=y_idx)
        true_transition_matrix = get_transition_matrix(ds_for_sst)
        if poly_degree:
            additional_input = [
                sst ** degree for degree in range(1, poly_degree + 1)
            ]
        else:
            additional_input = []
        modeled_transition_matrix = get_transition_matrix_from_model(
            model, additional_input
        )
        errors.append(((
            true_transition_matrix - modeled_transition_matrix
        ) ** 2).mean())
    return np.array(errors).mean()


def separate_model_for_each_sst(x_data, y_data):
    accuracies = []
    for sst in x_data.sst_1.unique():
        x_train, x_test, y_train, y_test = train_test_split(
            x_data[x_data.sst_1 == sst].drop('sst_1', axis=1),
            y_data[x_data.sst_1 == sst],
            test_size=0.5)
        model.fit(x_train, y_train)
        print(f'\n\nTrain Accuracy: {model.score(x_train, y_train)}')
        print(f'Test Accuracy: {model.score(x_test, y_test)}')
        print(model.coef_.round(2))
        accuracies.append(model.score(x_test, y_test))
    print('Overall accuracy: {}'.format(np.array(accuracies).mean()))


def train_transition_matrix_model(proportion_of_data=.5):
    x_data, y_data = format_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=1 - proportion_of_data)
    model.fit(x_train, y_train)
    return model


def get_transition_matrix_by_y_index():
    model = train_transition_matrix_model()
    transition_matrix_by_y_index = {}
    for y_idx, sst in enumerate(normalize_array(ds.SST.values[0, :, 0])):
        if poly_degree:
            additional_input = [
                sst ** degree for degree in range(1, poly_degree + 1)
            ]
        else:
            additional_input = []
        modeled_transition_matrix = get_transition_matrix_from_model(
            model, additional_input
        )
        transition_matrix_by_y_index[y_idx] = modeled_transition_matrix
    return transition_matrix_by_y_index


def get_true_transition_matrix_by_y_index():
    transition_matrix_by_y_index = {}
    for y_idx, sst in enumerate(normalize_array(ds.SST.values[0, :, 0])):
        ds_for_sst = ds.isel(y=y_idx)
        true_transition_matrix = get_transition_matrix(ds_for_sst)
        transition_matrix_by_y_index[y_idx] = true_transition_matrix
    return transition_matrix_by_y_index


# def get_transition_matrix_from_input_func():
#     x_data, y_data = format_training_data()
#     model.fit(x_data,)


if __name__ == '__main__':
    # print(f'\n\nPolynomial Degree: {poly_degree}')
    # x_data, y_data = format_training_data()
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_data, y_data, test_size=0.9)
    # model.fit(x_train, y_train)
    # get_modeled_sst_specific_transition_matrices(model)
    for poly_degree in range(5):
        print(f'\n\nPolynomial Degree: {poly_degree}')
        x_data, y_data = format_training_data()
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.9)
        model.fit(x_train, y_train)
        mse = get_transition_matrix_mse(model)
        print('Transition matrix MSE: {}'.format(mse))
    # print(f'Train Accuracy: {model.score(x_train, y_train)}')
    # print(f'Test Accuracy: {model.score(x_test, y_test)}')
    # separate_model_for_each_sst(x_data, y_data)
