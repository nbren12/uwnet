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
                    np.hstack(data.sel(data_to_select)[
                        ['QT', 'SLI']].to_array().values),
                    data.sel(data_to_select)[
                        ['LHF', 'SHF', 'SOLIN']].to_array().values.T
                ]))
                nn_output = mlp.forward(to_predict_from)
                q1_pred.append(nn_output[0][34:].detach().numpy())
                q2_pred.append(nn_output[0][:34].detach().numpy())
                data_to_select['time'] += dt
                true_state = np.hstack(
                    data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
                if include_known_forcing:
                    q_true = ((true_state - last_true_state) / (
                        dt * 86400)) - fqt_fsli
                    fqt_fsli = np.hstack(data.sel(
                        data_to_select)[['FQT', 'FSLI']].to_array().values)
                else:
                    q_true = (true_state - last_true_state) / (dt * 86400)
                q1_true.append(q_true[0, 34:])
                q2_true.append(q_true[0, :34])
                last_true_state = true_state
            q1_true = np.stack(q1_true)
            q2_true = np.stack(q2_true)
            q1_pred = np.stack(q1_pred)
            q2_pred = np.stack(q2_pred)
            q1_r2 = get_weighted_r2_score(q1_true, q1_pred, data)
            q2_r2 = get_weighted_r2_score(q2_true, q2_pred, data)
            q1_r2s.append(q1_r2)
            q2_r2s.append(q2_r2)
    print(f'Q1 R2: {mean(q1_r2s)}')
    print(f'Q2 R2: {mean(q2_r2s)}')


def plot_q_vs_nn_output(
        mlp,
        data,
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
            np.hstack(data.sel(data_to_select)[
                ['QT', 'SLI']].to_array().values),
            data.sel(data_to_select)[
                ['LHF', 'SHF', 'SOLIN']].to_array().values.T
        ]))
        nn_output = mlp.forward(to_predict_from)
        q1_pred.append(nn_output[34:].detach().numpy())
        q2_pred.append(nn_output[:34].detach().numpy())
        data_to_select['time'] += dt
        true_state = np.hstack(
            data.sel(data_to_select)[['QT', 'SLI']].to_array().values)
        if include_known_forcing:
            q_true = ((true_state - last_true_state) / (dt * 86400)) - fqt_fsli
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
    q1_r2 = get_weighted_r2_score(q1_true, q1_pred, data)
    print(f'Q1 Weighted R2 score: {q1_r2}')
    q2_r2 = get_weighted_r2_score(q2_true, q2_pred, data)
    print(f'Q2 Weighted R2 score: {q2_r2}')
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
