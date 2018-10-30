from stewart_intro.generate_training_data import normalize_data
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import numpy as np
import pandas as pd

default_dt = 0.125
default_n_time_steps = 10
default_batch_size = 32
default_n_ephocs = 5


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(71, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 68)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_batches(
        data,
        batch_size=default_batch_size,
        n_time_steps=default_n_time_steps):
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
        data, batch, n_time_steps=default_n_time_steps, dt=default_dt):
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
    true_states_with_inputs = [
        np.array([
            np.concatenate([
                cell.QT.values,
                cell.SLI.values,
                [cell.LHF.values],
                [cell.SHF.values],
                [cell.SOLIN.values]
            ])
            for cell in batch_values
        ])
    ]
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
            true_states_with_inputs.append(np.array([
                np.concatenate([
                    cell.QT.values,
                    cell.SLI.values,
                    [cell.LHF.values],
                    [cell.SHF.values],
                    [cell.SOLIN.values]
                ])
                for cell in batch_values
            ]))
            fqts_fslis.append(np.array([
                np.concatenate([cell.FQT.values, cell.FSLI.values])
                for cell in batch_values
            ]))
    return (
        torch.tensor(torch.from_numpy(x_data), requires_grad=True),
        torch.tensor(torch.from_numpy(true_states[-1]), requires_grad=True),
        torch.tensor(
            torch.from_numpy(np.array(fqts_fslis)), requires_grad=True),
        torch.tensor(true_states, requires_grad=True),
        torch.tensor(true_states_with_inputs, requires_grad=True),
    )


def squared_loss(y_hat, y):
    return (y - y_hat).pow(2)


def compute_error(
        mlp,
        x_data,
        targets,
        fqts_fslis,
        dt=default_dt,
        n_time_steps=default_n_time_steps):
    error = 0
    prediction = x_data[:, :-3]
    additional_variables = x_data[:, -3:]
    to_predict_from = x_data
    for i in range(n_time_steps):
        prediction = mlp.forward(to_predict_from)
        error = error + squared_loss(prediction, targets[i]).mean()
        if i != n_time_steps - 1:
            to_predict_from = torch.cat((prediction, additional_variables), 1)
    return error


def train_model(
        n_epochs=default_n_ephocs,
        dt=default_dt,
        n_time_steps=default_n_time_steps):
    data = normalize_data()
    batches = get_batches(data, n_time_steps=n_time_steps)
    mlp = MLP()
    n_batches = len(batches)
    test_batches = pd.DataFrame(
        np.concatenate(random.sample(
            batches, 50)), columns=batches[0].columns)
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    x_data_test, target_test, fqts_fslis_test, target_tests, _ = \
        get_formatted_training_data_from_batch(data, test_batches, dt=dt)
    for epoch_num in range(1, n_epochs + 1):
        for idx, batch in enumerate(batches):
            if not idx % 1000:
                err_est = compute_error(
                    mlp,
                    x_data_test,
                    target_tests,
                    fqts_fslis_test,
                    dt=dt,
                    n_time_steps=n_time_steps)
                print(f'{idx} of {n_batches} for epoch {epoch_num}')
                print(f'Total Error: {err_est}')
            x_data, target, fqts_fslis, targets, _ = \
                get_formatted_training_data_from_batch(data, batch, dt=dt)
            optimizer.zero_grad()
            loss = compute_error(
                mlp,
                x_data,
                targets,
                fqts_fslis,
                dt=dt,
                n_time_steps=n_time_steps)
            loss.backward()
            optimizer.step()    # Does the update
    return mlp


if __name__ == '__main__':
    train_model()
