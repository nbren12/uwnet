from stewart_intro.generate_training_data import normalize_data
import numpy as np
import pandas as pd
from itertools import product
import torch
from torch.autograd import Variable
import torch.nn.functional as F


n_hidden_nodes = 250
batch_size = 20
n_epochs = 1
n_time_steps = 1
dt = 0.125  # 3 hours in days
training_rate = .0001


def get_batches(data):
    time_values = data.isel(time=slice(0, len(data.time) - 1)).time.values
    points = np.array(
        [row for row in product(data.x.values, data.y.values, time_values)])
    np.random.shuffle(points)
    points = pd.DataFrame(points, columns=['x', 'y', 'time'])
    n_batches = int(len(points) / batch_size)
    batches = np.array_split(points, n_batches)
    return batches


def get_formatted_training_data_from_batch(data, batch):
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
    for i in range(1, n_time_steps + 1):
        batch_values = [
            data.sel(x=row.x, y=row.y, time=int(row.time + (dt * i)))
            for _, row in batch.iterrows()
        ]
        if i == n_time_steps:
            target = np.array([
                np.concatenate([cell.QT.values, cell.SLI.values])
                for cell in batch_values
            ])
        if i != n_time_steps:
            fqts_fslis.append(np.array([
                np.concatenate([cell.FQT.values, cell.FSLI.values])
                for cell in batch_values
            ]))
    return (
        Variable(torch.from_numpy(x_data), requires_grad=True),
        Variable(torch.from_numpy(target), requires_grad=True),
        Variable(torch.from_numpy(np.array(fqts_fslis)), requires_grad=True)
    )


def initialize_weights(data):
    w1 = Variable(
        (torch.rand((len(data.z) * 2) + 3, n_hidden_nodes) - .5) / 1000,
        requires_grad=True)
    w2 = Variable((torch.rand(n_hidden_nodes, len(data.z) * 2) - .5) / 1000,
                  requires_grad=True)
    return w1, w2


def predict(w1, w2, n_time_steps, x_data, fqts_fslis):
    prediction = x_data[:, :-3]
    for i in range(n_time_steps):
        prediction = prediction + (dt * fqts_fslis[i])
        layer_one_out = F.relu(x_data.matmul(w1))
        prediction = layer_one_out.matmul(w2)
        x_data[:, :-3] = prediction
    return prediction


def squared_loss(y_hat, y):
    return (1 / 2) * (y - y_hat).pow(2)


def compute_error(w1, w2, x_data, target, fqts_fslis):
    if len(x_data.size()) == 1:
        n = 1
    else:
        n = x_data.size()[0]
    y_hat = predict(w1, w2, n_time_steps, x_data, fqts_fslis)
    return ((1 / n) * squared_loss(
        y_hat, target).sum())


def train_model():
    data = normalize_data()
    batches = get_batches(data)
    w1, w2 = initialize_weights(data)
    for _ in range(n_epochs):
        for batch in batches:
            x_data, target, fqts_fslis = \
                get_formatted_training_data_from_batch(data, batch)
            err_est = compute_error(w1, w2, x_data, target, fqts_fslis)
            print(float(err_est))
            err_est.backward()
            w1.data = w1.data - (training_rate * w1.grad.data)
            w1.grad.data.zero_()
            w2.data = w2.data - (training_rate * w2.grad.data)
            w2.grad.data.zero_()


if __name__ == '__main__':
    train_model()
