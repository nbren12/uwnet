import math
import pickle
import torch
import pandas as pd
import numpy as np
from statistics import mean
from sklearn.metrics import r2_score
from stewart_intro.utils import project_dir
from stewart_intro.generate_training_data import normalize_data
from stewart_intro.simple_nn import (
    get_formatted_training_data_from_batch,
    default_dt,
)
import torch.nn.functional as F
from matplotlib import pyplot as plt


def load_nn():
    with open(
            project_dir +
            'stewart_intro/models/' +
            # 'simple_nn_99_batch_size_50000_iters.pkl', 'rb') as f:
            # 'simple_nn_100_1000_52345.pkl', 'rb') as f:
            'multi_step_nn_32_10_500_50000.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def plot_prognostic_predictions(w1, w2, data, dt=default_dt):
    x = data.x.values[35]
    y = data.y.values[3]
    n_time_steps = 100
    time = data.time.values[100]
    point = pd.DataFrame([[x, y, time]], columns=['x', 'y', 'time'])
    x_data, target, fqts_fslis, true_states, _ = \
        get_formatted_training_data_from_batch(
            data,
            point,
            n_time_steps=n_time_steps)
    true_states = np.stack(true_states.detach().numpy())[:, 0, :]
    to_predict_from = x_data
    qt_estimates = []
    sli_estimates = []
    qt_true = true_states[:, :34]
    sli_true = true_states[:, 34:]

    prediction = x_data[:, :-3]
    for i in range(n_time_steps):
        to_predict_from = torch.transpose(torch.cat(
            (
                torch.transpose(prediction, 0, 1),
                torch.transpose(x_data[:, -3:], 0, 1),
            )
        ), 0, 1)
        layer_one_out = F.relu(to_predict_from.matmul(w1))
        nn_output = layer_one_out.matmul(w2)
        prediction = prediction + (dt * (nn_output + fqts_fslis[i]))
        qt_estimates.append(prediction.detach().numpy()[0][:34])
        sli_estimates.append(prediction.detach().numpy()[0][34:])
    qt_estimates = np.stack(qt_estimates)
    sli_estimates = np.stack(sli_estimates)
    plt.contourf(qt_true.T)
    plt.colorbar()
    plt.title('True Normalized QT')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.show()
    plt.contourf(qt_estimates.T)
    plt.colorbar()
    plt.title('Predicted Normalized QT')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.show()
    plt.contourf(sli_true.T)
    plt.colorbar()
    plt.title('True Normalized SLI')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.show()
    plt.contourf(sli_estimates.T)
    plt.colorbar()
    plt.title('Predicted Normalized SLI')
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.show()


def get_weighted_r2_score(true, pred, data):
    weights = np.concatenate([data.layer_mass.values] * true.shape[0])
    return r2_score(true.ravel(), pred.ravel(), sample_weight=weights)


def get_diagnostic_r2_score(w1, w2, data, dt=default_dt):
    time_steps = 100
    q1_r2s = []
    q2_r2s = []
    for x in np.random.choice(data.x.values, 10):
        for y in np.random.choice(data.y.values, 10):
            for time in np.random.choice(
                    data.time.values[data.time.values < data.time.values[
                        -time_steps]], 3):
                point = pd.DataFrame(
                    [[x, y, time]], columns=['x', 'y', 'time'])
                (
                    x_data,
                    target,
                    fqts_fslis,
                    true_states,
                    true_states_with_inputs
                ) = get_formatted_training_data_from_batch(
                    data,
                    point,
                    n_time_steps=time_steps,
                    dt=dt)
                true_states = np.stack(true_states.detach().numpy())[:, 0, :]
                q1_true = []
                q2_true = []
                q1_pred = []
                q2_pred = []
                last_true_state = x_data[:, :-3].detach().numpy()
                for idx in range(time_steps):
                    to_predict_from = true_states_with_inputs[idx]
                    layer_one_out = F.relu(to_predict_from.matmul(w1))
                    nn_output = layer_one_out.matmul(w2)
                    q1_pred.append(nn_output[0][34:].detach().numpy())
                    q2_pred.append(nn_output[0][:34].detach().numpy())
                    q_true = ((true_states[idx] -
                               last_true_state) / dt) - fqts_fslis[
                        idx].detach().numpy()
                    q1_true.append(q_true[0, 34:])
                    q2_true.append(q_true[0, :34])
                    last_true_state = true_states[idx]
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


def plot_q_vs_nn_output(w1, w2, data, save_location=None, dt=default_dt):
    x = data.x.values[35]
    y = data.y.values[3]
    time_steps = 100
    time = data.time.values[100]
    point = pd.DataFrame([[x, y, time]], columns=['x', 'y', 'time'])
    x_data, target, fqts_fslis, true_states, true_states_with_inputs = \
        get_formatted_training_data_from_batch(
            data,
            point,
            n_time_steps=time_steps)
    true_states = np.stack(true_states.detach().numpy())[:, 0, :]
    q1_true = []
    q2_true = []
    q1_pred = []
    q2_pred = []
    last_true_state = x_data[:, :-3].detach().numpy()
    for idx in range(time_steps):
        to_predict_from = true_states_with_inputs[idx]
        layer_one_out = F.relu(to_predict_from.matmul(w1))
        nn_output = layer_one_out.matmul(w2)
        q1_pred.append(nn_output[0][34:].detach().numpy())
        q2_pred.append(nn_output[0][:34].detach().numpy())
        q_true = ((true_states[idx] - last_true_state) / dt) - fqts_fslis[
            idx].detach().numpy()
        q1_true.append(q_true[0, 34:])
        q2_true.append(q_true[0, :34])
        last_true_state = true_states[idx]
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
    if save_location:
        plt.savefig(save_location)
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
    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()


if __name__ == '__main__':
    data = normalize_data(filter_down=True)
    w1, w2 = load_nn()
    # plot_q_vs_nn_output(w1, w2, data)
    plot_prognostic_predictions(w1, w2, data)
