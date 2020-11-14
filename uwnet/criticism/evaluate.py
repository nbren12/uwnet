import torch
from uwnet.loss import get_input_output
from uwnet.utils import mean_other_dims
import json
from tqdm import tqdm
import argparse

import uwnet.ml_models.nn.datasets_handler as d


def batch_to_residual(model, batch):
    from uwnet.timestepper import Batch

    batch = Batch(batch.float(), prognostics=["QT", "SLI"])
    with torch.no_grad():
        prediction, truth = get_input_output(model, 0.125, batch)
    return prediction - truth


def vertically_resolved_mse_from_residual(residual):
    return {k: mean_other_dims(residual[k] ** 2, 2).squeeze() for k in residual}


def batch_to_mse(model, batch):
    residual = batch_to_residual(model, batch)
    return vertically_resolved_mse_from_residual(residual)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="path to zarr reshaped training or test data")
    parser.add_argument("model", help="path to model")

    return parser.parse_args()


args = _parse_args()

model_path = args.model
path = args.data
prognostics = ["QT", "SLI"]

model = torch.load(model_path)
train_dataset = d.get_dataset(path, predict_radiation=False)
dl = d.get_data_loader(train_dataset, prognostics, batch_size=64)

total = {}
count = 0
for batch in tqdm(dl):
    mse = batch_to_mse(model, batch)
    for key in mse:
        count += 1
        alpha = 1 / count
        zeros = torch.zeros_like(mse[key])
        total[key] = total.get(key, zeros) * (1 - alpha) + mse[key] * alpha

total = {"mse_apparent_source_" + k.lower(): total[k].numpy().tolist() for k in total}
print(json.dumps(total))
