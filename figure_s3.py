import glob
import re
import json
import pandas as pd
from toolz import assoc
import xarray as xr

tarball = "~/data/uwnet/2020-08-14-7d11224e5867446011a9c16b4eda9fd32a23e6b2-nn.tar"

pattern = re.compile(r"nn\/(\w+)\/(\d+)\.(\w+)\.json$")
metrics = glob.glob("nn/**/*.json")

data = []


def _open_all_metrics(metrics):
    for path in metrics:
        m = pattern.search(path)
        if m:
            model, epoch, train_or_test = m.groups()
            epoch = int(epoch)

            record = {
                "model": model,
                "epoch": epoch,
                "train_or_test": train_or_test,
            }

            with open(path) as f:
                try:
                    value = json.load(f)
                except json.decoder.JSONDecodeError:
                    pass
                else:
                    for k, v in value.items():
                        for n, it in enumerate(v):
                            yield {"name": k, "n": n, "value": it, **record}


def read_metrics(files: str) -> xr.Dataset:
    metrics = glob.glob(files, recursive=True)
    df = pd.DataFrame.from_records(_open_all_metrics(metrics)).set_index(
        ["name", "model", "n", "epoch", "train_or_test"]
    ).unstack(level=0)["value"]
    return xr.Dataset.from_dataframe(df)

import seaborn
import matplotlib.pyplot as plt

ds = read_metrics("/Users/noah/workspace/uwnet/nn/**/*.json")
df = ds.mean('n').to_dataframe().reset_index()
plt.figure()
seaborn.lineplot(x='epoch', y='mse_apparent_source_qt', hue='train_or_test', data=df, ci=None)
plt.figure()
seaborn.lineplot(x='epoch', y='mse_apparent_source_sli', hue='train_or_test', data=df, ci=None)
plt.show()
