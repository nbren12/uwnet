from stewart_intro.generate_training_data import normalize_data
from stewart_intro.utils import pickle_model
import numpy as np
import pandas as pd
from itertools import product
import random
import torch
import torch.nn.functional as F


n_hidden_nodes = 10000
batch_size = 100
n_epochs = 100
n_time_steps = 10
dt = 0.125  # 3 hours in days
training_rate = 0.01


def get_batches(data, batch_size=batch_size):
    time_values = data.isel(
        time=slice(0, len(data.time) - n_time_steps)).time.values
    points = np.array(
        [row for row in product(data.x.values, data.y.values, time_values)])
    np.random.shuffle(points)
    points = pd.DataFrame(points, columns=['x', 'y', 'time'])
    n_batches = int(len(points) / batch_size)
    batches = np.array_split(points, n_batches)
    return batches


def get_formatted_training_data_from_batch(
        data, batch, n_time_steps=n_time_steps):
    batch_values = [
        data.sel(x=row.x, y=row.y, time=row.time)
        for _, row in batch.iterrows()
    ]
    x_data = np.array([
        np.concatenate(
            [
                cell.QT.values,
                cell.SLI.values,
                [cell.LHF.values],
                [cell.SHF.values],
                [cell.SOLIN.values]
            ]
        ) for cell in batch_values
    ])
    fqts_fslis = [
        np.array(
            [np.concatenate([cell.FQT.values, cell.FSLI.values])
             for cell in batch_values])
    ]
    true_states = []
    for i in range(1, n_time_steps + 1):
        batch_values = [
            data.sel(x=row.x, y=row.y, time=row.time + (dt * i))
            for _, row in batch.iterrows()
        ]
        true_states.append(np.array([
            np.concatenate([cell.QT.values, cell.SLI.values])
            for cell in batch_values
        ]))
        if i != n_time_steps:
            fqts_fslis.append(np.array([
                np.concatenate([cell.FQT.values, cell.FSLI.values])
                for cell in batch_values
            ]))
    return (
        torch.tensor(torch.from_numpy(x_data), requires_grad=True),
        torch.tensor(torch.from_numpy(true_states[-1]), requires_grad=True),
        torch.tensor(
            torch.from_numpy(np.array(fqts_fslis)), requires_grad=True),
        torch.tensor(true_states, requires_grad=True)
    )


def initialize_weights(data):
    w1 = torch.tensor((torch.rand(
        (len(data.z) * 2) + 3, n_hidden_nodes) - .5) / 1000,
        requires_grad=True)
    w2 = torch.tensor((torch.rand(
        n_hidden_nodes, len(data.z) * 2) - .5) / 1000, requires_grad=True)
    return w1, w2


def predict(w1, w2, n_time_steps, x_data, fqts_fslis):
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
    return prediction


def squared_loss(y_hat, y):
    return (y - y_hat).pow(2)


def compute_error(w1, w2, x_data, targets, fqts_fslis):
    error = 0
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
        error = error + squared_loss(prediction, targets[i])
    return error.mean()


def train_model():
    data = normalize_data()
    iter_ = 0
    batches = get_batches(data)
    w1, w2 = initialize_weights(data)
    v1, v2 = initialize_weights(data)
    n_batches = len(batches)
    test_batches = pd.DataFrame(
        np.concatenate(random.sample(
            batches, 500)), columns=batches[0].columns)
    x_data_test, target_test, fqts_fslis_test, target_tests = \
        get_formatted_training_data_from_batch(data, test_batches)
    for epoch_num in range(1, n_epochs + 1):
        for idx, batch in enumerate(batches):
            if not idx % 1000:
                err_est = compute_error(
                    w1, w2, x_data_test, target_tests, fqts_fslis_test)
                print(f'{idx} of {n_batches} for epoch {epoch_num}')
                print(f'Total Error: {err_est}')
            x_data, target, fqts_fslis, targets = \
                get_formatted_training_data_from_batch(data, batch)
            err_est = compute_error(
                w1 - 0.9 * v1, w2 - 0.9 * v2, x_data, targets, fqts_fslis)
            err_est.backward()

            # v1.data = (0.9 * v1.data) + ((training_rate / (
            #     (iter_ + 1) ** (2 / 3))) * (
            #     w1.grad.data - (0.9 * v1.grad.data)))
            # v2.data = (0.9 * v2.data) + ((training_rate / (
            #     (iter_ + 1) ** (2 / 3))) * (
            #     w2.grad.data - (0.9 * v2.grad.data)))
            v1.data = (0.9 * v1.data) + (training_rate * (
                w1.grad.data - (0.9 * v1.grad.data)))
            v2.data = (0.9 * v2.data) + (training_rate * (
                w2.grad.data - (0.9 * v2.grad.data)))

            w1.data = w1.data - v1.data
            w2.data = w2.data - v2.data
            w1.grad.data.zero_()
            w2.grad.data.zero_()
            v1.grad.data.zero_()
            v2.grad.data.zero_()
            iter_ += 1
            if not idx % 10000 and idx != 0:
                pickle_model(
                    [w1, w2], 'multi_step_nn_{}_{}_{}_{}'.format(
                        batch_size,
                        n_time_steps, n_hidden_nodes, idx + (
                            (epoch_num - 1) * 10000)))
        pickle_model(
            [w1, w2], 'multi_step_nn_{}_{}_{}_{}'.format(
                batch_size,
                n_time_steps, n_hidden_nodes, idx + ((epoch_num - 1) * 10000)))
    return w1, w2


if __name__ == '__main__':
    train_model()
