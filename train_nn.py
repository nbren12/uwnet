from copy import deepcopy
from stewart_intro.generate_training_data import normalize_data
from stewart_intro.utils import pickle_model
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

default_dt = 0.125
min_dt = 0.125
default_n_time_steps = 1
default_batch_size = 32
default_n_ephocs = 1
default_training_rate = 0.001
default_include_known_forcing = True
default_n_hidden_nodes = 500
seconds_per_day = 86400


class MLP(nn.Module):

    def __init__(self, n_hidden_nodes=default_n_hidden_nodes):
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
        n_time_steps=default_n_time_steps,
        dt=default_dt):
    time_values = data.time.values[
        data.time.values < (
            data.time.values.max() - dt * n_time_steps)]
    np.random.shuffle(time_values)
    time_batches = np.array_split(time_values, batch_size)
    batches = np.array([
        {'x': x, 'y': y, 'time': time_batch}
        for x in data.x.values
        for y in data.y.values
        for time_batch in time_batches
    ])
    np.random.shuffle(batches)
    return batches


def squared_loss(y_hat, y):
    return (y - y_hat).pow(2)


def compute_error(
        mlp,
        data,
        batch,
        layer_mass,
        normalization_dict,
        dt=default_dt,
        n_time_steps=default_n_time_steps,
        include_known_forcing=default_include_known_forcing):
    error = 0
    qt_sli = np.hstack(data.sel(batch)[['QT', 'SLI']].to_array().values)
    additional_predictors = data.sel(
        batch)[[
            'LHF_normalized',
            'SHF_normalized',
            'SOLIN_normalized']].to_array().values.T
    to_predict_from = torch.tensor(
        np.hstack([(qt_sli - normalization_dict['qt_sli']['mean']) / (
            normalization_dict['qt_sli']['sd']),
            additional_predictors]), requires_grad=True)
    additional_predictors = torch.tensor(
        additional_predictors, requires_grad=True)
    prediction = torch.tensor(qt_sli, requires_grad=True)
    target_locations = deepcopy(batch)
    for i in range(n_time_steps):
        nn_output = mlp.forward(to_predict_from)
        if include_known_forcing:
            fqt_fsli = torch.tensor(
                np.hstack(
                    data.sel(target_locations)[
                        ['FQT', 'FSLI']].to_array().values),
                requires_grad=True)
            fqt_fsli_divisor = 1
            for dt_step in range(int(dt / min_dt)):
                target_locations['time'] += min_dt
                fqt_fsli = fqt_fsli + torch.tensor(
                    np.hstack(
                        data.sel(target_locations)[
                            ['FQT', 'FSLI']].to_array().values),
                    requires_grad=True)
                fqt_fsli_divisor += 1
            prediction = prediction + (
                (fqt_fsli / fqt_fsli_divisor) * (dt * seconds_per_day))
        else:
            target_locations['time'] += dt
        prediction = prediction + nn_output
        target = torch.tensor(np.hstack(
            data.sel(target_locations)[['QT', 'SLI']].to_array().values),
            requires_grad=True)
        error = error + (squared_loss(prediction, target).mean(0).dot(
            layer_mass) / layer_mass.sum())
        if i != n_time_steps - 1:
            to_predict_from = torch.cat(
                ((prediction - normalization_dict['qt_sli']['mean']) / (
                    normalization_dict['qt_sli']['sd']),
                    additional_predictors), 1)
    return error


def get_evaluation_batches(batches):
    batch_sample = random.sample(batches.tolist(), 50)
    seed = deepcopy(np.random.choice(batches))
    seed['time'] = np.concatenate([
        batch['time'] for batch in batch_sample
    ])
    return seed


def estimate_total_mse(
        mlp,
        data,
        batches,
        layer_mass,
        normalization_dict,
        dt,
        n_time_steps,
        include_known_forcing):
    random.seed(20)
    batch_sample = random.sample(batches.tolist(), 1000)
    total_loss = 0
    for batch in batch_sample:
        loss = compute_error(
            mlp,
            data,
            batch,
            layer_mass,
            normalization_dict,
            dt=dt,
            n_time_steps=n_time_steps,
            include_known_forcing=include_known_forcing)
        total_loss += loss.detach().numpy()
    print(f'Total MSE: {total_loss / len(batch_sample)}')
    return total_loss / len(batch_sample)


def train_model(
        n_epochs=default_n_ephocs,
        dt=default_dt,
        n_time_steps=default_n_time_steps,
        batch_size=default_batch_size,
        model_name=None,
        training_rate=default_training_rate,
        include_known_forcing=default_include_known_forcing,
        n_hidden_nodes=default_n_hidden_nodes,
        verbose=False):
    if not model_name:
        model_name = 'multi_step_nn_{}_{}'.format(
            batch_size, n_time_steps) + '_{}'
    data, normalization_dict = normalize_data()
    normalization_dict['qt_sli'] = {
        'mean': np.concatenate([
            normalization_dict['QT']['mean'],
            normalization_dict['SLI']['mean']
        ]),
        'sd': np.array([
            normalization_dict['QT']['sd']] * 34 +
            [normalization_dict['SLI']['sd']] * 34)
    }
    w = data.layer_mass.values / data.layer_mass.values.mean()
    layer_mass = torch.cat([
        torch.tensor(w, requires_grad=True),
        torch.tensor(w, requires_grad=True)
    ]).float()
    batches = get_batches(
        data, n_time_steps=n_time_steps, batch_size=batch_size, dt=dt)
    mlp = MLP(n_hidden_nodes=n_hidden_nodes)
    n_batches = len(batches)
    evaluation_batch = get_evaluation_batches(batches)
    optimizer = optim.Adam(mlp.parameters(), lr=training_rate)
    for epoch_num in range(1, n_epochs + 1):
        for idx, batch in enumerate(batches):
            if verbose and not idx % 1000:
                err_est = compute_error(
                    mlp,
                    data,
                    evaluation_batch,
                    layer_mass,
                    normalization_dict,
                    dt=dt,
                    n_time_steps=n_time_steps,
                    include_known_forcing=include_known_forcing)
                print(f'{idx} of {n_batches} for epoch {epoch_num}')
                print(f'Total Error: {err_est}')
            optimizer.zero_grad()
            loss = compute_error(
                mlp,
                data,
                batch,
                layer_mass,
                normalization_dict,
                dt=dt,
                n_time_steps=n_time_steps,
                include_known_forcing=include_known_forcing)
            loss.backward()
            optimizer.step()    # Does the update
        pickle_model(mlp, model_name.format(idx + ((epoch_num - 1) * 10000)))
    total_mse = estimate_total_mse(
        mlp,
        data,
        batches,
        layer_mass,
        normalization_dict,
        dt,
        n_time_steps,
        include_known_forcing)
    return mlp, data, normalization_dict, total_mse


if __name__ == '__main__':
    mlp, data, normalization_dict, total_mse = train_model()
