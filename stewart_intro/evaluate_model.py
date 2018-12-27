import pandas as pd
import torch
import math
import numpy as np
from statistics import mean
from sklearn.metrics import r2_score
from train_nn import (
    default_dt,
    default_include_known_forcing,
)
from matplotlib import pyplot as plt


def get_weighted_r2_score(true, pred, data):
    weights = np.concatenate([data.layer_mass.values] * true.shape[0])
    return r2_score(true.ravel(), pred.ravel(), sample_weight=weights)


def get_diagnostic_r2_score(
        mlp,
        data,
        normalization_dict,
        dt=default_dt,
        include_known_forcing=default_include_known_forcing):
    time_steps = 50
    q1_r2s = []
    q2_r2s = []
    np.random.seed(33)
    for x in np.random.choice(data.x.values, 10):
        for y in np.random.choice(data.y.values, 10):
            times = np.random.choice(
                data.time.values[data.time.values < data.time.values[
                    -(time_steps * int(dt / default_dt))]], 6)
            data_to_select = {'x': x, 'y': y, 'time': times}
            q1_true = []
            q2_true = []
            q1_pred = []
            q2_pred = []
            last_true_state = np.hstack(
                data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
            fqt_fsli = np.hstack(
                data.sel(data_to_select)[['FQT', 'FSLI']].to_array().values)
            for idx in range(time_steps):
                to_predict_from = torch.tensor(np.hstack([
                    (np.hstack(data.sel(data_to_select)[
                        ['QT', 'SLI']].to_array().values) -
                        normalization_dict['qt_sli']['mean']) /
                    normalization_dict['qt_sli']['sd'],
                    data.sel(data_to_select)[
                        ['LHF_normalized',
                         'SHF_normalized',
                         'SOLIN_normalized']].to_array().values.T
                ]))
                nn_output = mlp.forward(to_predict_from)
                q1_pred.append(nn_output.detach().numpy()[:, 34:])
                q2_pred.append(nn_output.detach().numpy()[:, :34])
                data_to_select['time'] += dt
                true_state = np.hstack(
                    data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
                if include_known_forcing:
                    q_true = (true_state - last_true_state - (
                              dt * 86400 * fqt_fsli))
                    fqt_fsli = np.hstack(data.sel(
                        data_to_select)[['FQT', 'FSLI']].to_array().values)
                else:
                    q_true = (true_state - last_true_state) / (dt * 86400)
                q1_true.append(q_true[:, 34:])
                q2_true.append(q_true[:, :34])
                last_true_state = true_state
            q1_true = np.vstack(q1_true)
            q2_true = np.vstack(q2_true)
            q1_pred = np.vstack(q1_pred)
            q2_pred = np.vstack(q2_pred)
            q1_r2 = get_weighted_r2_score(q1_true, q1_pred, data)
            q2_r2 = get_weighted_r2_score(q2_true, q2_pred, data)
            q1_r2s.append(q1_r2)
            q2_r2s.append(q2_r2)
    print(f'Q1 R2: {mean(q1_r2s)}')
    print(f'Q2 R2: {mean(q2_r2s)}')


def draw_histogram(values, bins=40, x_min=None, x_max=None,
                   x_label='', y_label='Counts', title='',
                   figsize=None,
                   show=True, save_to_filepath=None):
    if x_max is None:
        x_max = max(values)
    if x_min is None:
        x_min = min(values)
    if figsize is not None:
        plt.figure(figsize=figsize)
    n_, bins, patches = plt.hist(values, bins=bins)
    plt.plot(bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.axis([x_min, x_max, 0, max(n_)])
    plt.grid(True)
    if save_to_filepath:
        plt.savefig(save_to_filepath)
    if show:
        plt.show()
    plt.close()


def plot_residuals_and_get_correlation_matrix(
        mlp,
        data,
        batches,
        layer_mass,
        normalization_dict,
        dt,
        n_time_steps,
        include_known_forcing):
    time_steps = 50
    np.random.seed(33)
    q1_trues = []
    q1_preds = []
    q2_trues = []
    q2_preds = []
    for x in np.random.choice(data.x.values, 10):
        for y in np.random.choice(data.y.values, 10):
            times = np.random.choice(
                data.time.values[data.time.values < data.time.values[
                    -(time_steps * int(dt / default_dt))]], 6)
            data_to_select = {'x': x, 'y': y, 'time': times}
            q1_true = []
            q2_true = []
            q1_pred = []
            q2_pred = []
            last_true_state = np.hstack(
                data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
            fqt_fsli = np.hstack(
                data.sel(data_to_select)[['FQT', 'FSLI']].to_array().values)
            for idx in range(time_steps):
                to_predict_from = torch.tensor(np.hstack([
                    (np.hstack(data.sel(data_to_select)[
                        ['QT', 'SLI']].to_array().values) -
                        normalization_dict['qt_sli']['mean']) /
                    normalization_dict['qt_sli']['sd'],
                    data.sel(data_to_select)[
                        ['LHF_normalized',
                         'SHF_normalized',
                         'SOLIN_normalized']].to_array().values.T
                ]))
                nn_output = mlp.forward(to_predict_from)
                q1_pred.append(nn_output.detach().numpy()[:, 34:])
                q2_pred.append(nn_output.detach().numpy()[:, :34])
                data_to_select['time'] += dt
                true_state = np.hstack(
                    data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
                if include_known_forcing:
                    q_true = (true_state - last_true_state - (
                              dt * 86400 * fqt_fsli))
                    fqt_fsli = np.hstack(data.sel(
                        data_to_select)[['FQT', 'FSLI']].to_array().values)
                else:
                    q_true = (true_state - last_true_state) / (dt * 86400)
                q1_true.append(q_true[:, 34:])
                q2_true.append(q_true[:, :34])
                last_true_state = true_state
            q1_true = np.vstack(q1_true)
            q2_true = np.vstack(q2_true)
            q1_pred = np.vstack(q1_pred)
            q2_pred = np.vstack(q2_pred)
            if len(q1_trues):
                q1_trues = np.concatenate([q1_trues, q1_true])
                q2_trues = np.concatenate([q2_trues, q2_true])
                q1_preds = np.concatenate([q1_preds, q1_pred])
                q2_preds = np.concatenate([q2_preds, q2_pred])
            else:
                q1_trues = q1_true
                q2_trues = q2_true
                q1_preds = q1_pred
                q2_preds = q2_pred
    q1_residuals = q1_preds - q1_trues
    q2_residuals = q2_preds - q2_trues
    for name, residuals in [('Q1', q1_residuals), ('Q2', q2_residuals)]:
        for i in range(34):
            data = residuals[:, i]
            draw_histogram(
                data, x_label='Residual', y_label='Count',
                title=f'{name} Residuals for the Z-level {i+1}',
                show=False,
                save_to_filepath=f'/Users/stewart/Desktop/{name}_{i}_residuals.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(
        pd.DataFrame(q1_residuals).corr(), interpolation='nearest')
    fig.colorbar(cax)
    plt.title('Q1 Correlation Matrix')
    plt.savefig('/Users/stewart/Desktop/q1_correlation_matrix.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(
        pd.DataFrame(q2_residuals).corr(), interpolation='nearest')
    fig.colorbar(cax)
    plt.title('Q2 Correlation Matrix')
    plt.savefig('/Users/stewart/Desktop/q2_correlation_matrix.png')


def plot_q_vs_nn_output(
        mlp,
        data,
        normalization_dict,
        save_location_format_str=None,
        dt=default_dt,
        include_known_forcing=default_include_known_forcing):
    time_steps = 50
    x = data.x.values[35]
    y = data.y.values[3]
    time = data.time.values[20]
    data_to_select = {'x': x, 'y': y, 'time': time}
    q1_true = []
    q2_true = []
    q1_pred = []
    q2_pred = []
    last_true_state = np.hstack(
        data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
    fqt_fsli = np.hstack(
        data.sel(data_to_select)[['FQT', 'FSLI']].to_array().values)
    for idx in range(time_steps):
        to_predict_from = torch.tensor(np.hstack([
            (np.hstack(data.sel(data_to_select)[
                ['QT', 'SLI']].to_array().values) -
                normalization_dict['qt_sli']['mean']) /
            normalization_dict['qt_sli']['sd'],
            data.sel(data_to_select)[
                ['LHF_normalized',
                 'SHF_normalized',
                 'SOLIN_normalized']].to_array().values.T
        ]))
        nn_output = mlp.forward(to_predict_from)
        q1_pred.append(nn_output[34:].detach().numpy())
        q2_pred.append(nn_output[:34].detach().numpy())
        data_to_select['time'] += dt
        true_state = np.hstack(
            data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
        if include_known_forcing:
            q_true = (true_state - last_true_state - (
                dt * 86400 * fqt_fsli))
            fqt_fsli = np.hstack(data.sel(
                data_to_select)[['FQT', 'FSLI']].to_array().values)
        else:
            q_true = (true_state - last_true_state) / (dt * 86400)
        q1_true.append(q_true[34:])
        q2_true.append(q_true[:34])
        last_true_state = true_state
    q1_true = np.stack(q1_true)
    q2_true = np.stack(q2_true)
    q1_pred = np.stack(q1_pred)
    q2_pred = np.stack(q2_pred)
    # q1_r2 = get_weighted_r2_score(q1_true, q1_pred, data)
    # print(f'Q1 Weighted R2 score: {q1_r2}')
    # q2_r2 = get_weighted_r2_score(q2_true, q2_pred, data)
    # print(f'Q2 Weighted R2 score: {q2_r2}')
    vmin = math.floor(min(q1_true.min(), q1_pred.min()))
    vmax = math.ceil(max(q1_true.max(), q1_pred.max())) + 1
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0].contourf(q1_true.T, vmin=vmin, vmax=vmax)
    axs[1].contourf(q1_pred.T, vmin=vmin, vmax=vmax)
    axs[0].set_title('True Normalized Q1')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Z')
    axs[1].set_title('Predicted Normalized Q1')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Z')
    plt.subplots_adjust(hspace=0.7)
    fig.colorbar(ax0, ax=axs.ravel().tolist())
    if save_location_format_str:
        plt.savefig(save_location_format_str.format('Q1'))
    else:
        plt.show()

    vmin = math.floor(min(q2_true.min(), q2_pred.min()))
    vmax = math.ceil(max(q2_true.max(), q2_pred.max())) + 1
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0].contourf(q2_true.T, vmin=vmin, vmax=vmax)
    axs[1].contourf(q2_pred.T, vmin=vmin, vmax=vmax)
    axs[0].set_title('True Normalized Q2')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Z')
    axs[1].set_title('Predicted Normalized Q2')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Z')
    plt.subplots_adjust(hspace=0.7)
    fig.colorbar(ax0, ax=axs.ravel().tolist())
    if save_location_format_str:
        plt.savefig(save_location_format_str.format('Q2'))
    else:
        plt.show()


def plot_model_error_output(
        mlp,
        data,
        normalization_dict,
        save_location_format_str=None,
        dt=default_dt,
        include_known_forcing=default_include_known_forcing):
    time_steps = 50
    x = data.x.values[35]
    y = data.y.values[3]
    time = data.time.values[20]
    data_to_select = {'x': x, 'y': y, 'time': time}
    qt_true = []
    sli_true = []
    qt_pred = []
    sli_pred = []
    prediction = np.hstack(
        data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
    fqt_fsli = np.hstack(
        data.sel(data_to_select)[['FQT', 'FSLI']].to_array().values)
    for idx in range(time_steps):
        to_predict_from = torch.tensor(np.hstack([
            (np.hstack(data.sel(data_to_select)[
                ['QT', 'SLI']].to_array().values) -
                normalization_dict['qt_sli']['mean']) /
            normalization_dict['qt_sli']['sd'],
            data.sel(data_to_select)[
                ['LHF_normalized',
                 'SHF_normalized',
                 'SOLIN_normalized']].to_array().values.T
        ]))
        nn_output = mlp.forward(to_predict_from).detach().numpy()
        if include_known_forcing:
            prediction = prediction + (dt * 86400 * fqt_fsli) + nn_output
        else:
            prediction = prediction + nn_output
        qt_pred.append(prediction[:34])
        sli_pred.append(prediction[34:])
        data_to_select['time'] += dt
        fqt_fsli = np.hstack(data.sel(
            data_to_select)[['FQT', 'FSLI']].to_array().values)
        true_state = np.hstack(
            data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
        qt_true.append(true_state[:34])
        sli_true.append(true_state[34:])
    qt_true = np.stack(qt_true)
    sli_true = np.stack(sli_true)
    qt_pred = np.stack(qt_pred)
    sli_pred = np.stack(sli_pred)
    # qt_r2 = get_weighted_r2_score(qt_true, qt_pred, data)
    # print(f'qt Weighted R2 score: {qt_r2}')
    # sli_r2 = get_weighted_r2_score(sli_true, sli_pred, data)
    # print(f'sli Weighted R2 score: {sli_r2}')
    vmin = math.floor(min(qt_true.min(), qt_pred.min()))
    vmax = math.ceil(max(qt_true.max(), qt_pred.max())) + 1
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0].contourf(qt_true.T, vmin=vmin, vmax=vmax)
    axs[1].contourf(qt_pred.T, vmin=vmin, vmax=vmax)
    axs[0].set_title('True QT')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Z')
    axs[1].set_title('Predicted QT')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Z')
    plt.subplots_adjust(hspace=0.7)
    fig.colorbar(ax0, ax=axs.ravel().tolist())
    if save_location_format_str:
        plt.savefig(save_location_format_str.format('Q1'))
    else:
        plt.show()

    vmin = math.floor(min(sli_true.min(), sli_pred.min()))
    vmax = math.ceil(max(sli_true.max(), sli_pred.max())) + 1
    fig, axs = plt.subplots(2, 1)
    ax0 = axs[0].contourf(sli_true.T, vmin=vmin, vmax=vmax)
    axs[1].contourf(sli_pred.T, vmin=vmin, vmax=vmax)
    axs[0].set_title('True SLI')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Z')
    axs[1].set_title('Predicted SLI')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Z')
    plt.subplots_adjust(hspace=0.7)
    fig.colorbar(ax0, ax=axs.ravel().tolist())
    if save_location_format_str:
        plt.savefig(save_location_format_str.format('Q2'))
    else:
        plt.show()
