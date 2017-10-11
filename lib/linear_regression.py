import os
import numpy as np
import xarray as xr
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union
from lib.sklearn import Select, Stacker, Weighter

def get_sizes_from_union(union, test_data):
    sizes = []
    for name, transformer in union.transformer_list:
        out = transformer.fit_transform(test_data)
        sizes.append(out.shape[1])
    return sizes

def compute_mat(mod, w, in_sizes, out_sizes):
    """Compute matrix for mca

    Warning this code is extremely brittle and should not be reused.
    """

    union = mod.named_steps['featureunion']
    in_offsets = np.cumsum(in_sizes)
    out_offsets = np.cumsum(out_sizes)

    in_splits = in_offsets[:-1]
    out_splits = out_offsets[:-1]

    # compute LRF
    n = in_offsets[-1]
    I = np.eye(n)
    lrf = mod.steps[-1][1].predict(I)
    lrf -= mod.steps[-1][1].predict(I*0)

    # unweight the output
    w = w.data
    W = np.sqrt(np.diag(1/np.hstack((w, w))))

    lrf = lrf @ W

    lrf_dict = {(out_key, in_key): in_val
                for out_key, out_val in zip(['Q1c', 'Q2'], np.split(lrf, out_splits, axis=1))
                for in_key, in_val in zip(['qt', 'sl', 'lhf', 'shf'], np.split(out_val, in_splits, axis=0))}

    return lrf_dict

mem = joblib.Memory("/tmp/mycache")

data3d = snakemake.input.data3d
data2d = snakemake.input.data2d
weight = snakemake.input.weight
model_file = snakemake.output[0]

D = xr.open_mfdataset(data3d)
D2 = xr.open_mfdataset(data2d)
w = xr.open_dataarray(weight)

D = D.merge(D2, join='inner')
D = D.assign(Q1c=D.Q1 - D.QRAD)

d_train, d_test = D.sel(time=slice(0, 50)), D.sel(time=slice(50, None))

# union
union = make_union(
    make_pipeline(Select('QT', sel={'z': slice(0, 10e3)}), Stacker(['z'])),
    make_pipeline(Select('SL'), Stacker(['z'])),
    make_pipeline(Select('SHF'), Stacker()),
    make_pipeline(Select('LHF'), Stacker()))

output_union = make_union(
    make_pipeline(
        Select('Q1c'), Weighter(np.sqrt(w)), Stacker(['z'])),
    make_pipeline(
        Select('Q2'), Weighter(np.sqrt(w)), Stacker(['z'])))

# get union sizes
in_sizes = get_sizes_from_union(union, d_train.isel(time=0))
out_sizes = get_sizes_from_union(output_union, d_train.isel(time=0))

mod = make_pipeline(union, LinearRegression())

y_test = output_union.fit_transform(d_test)
y_train = output_union.fit_transform(d_train)

x_train = union.fit_transform(d_train)
x_test = union.fit_transform(d_test)

print("Fitting linear regression model")
mod.fit(d_train, y_train)
print("Computing score")
score = mod.score(d_train, y_train)
print(f"Saving to {model_file}")

mat = compute_mat(mod, w, in_sizes=in_sizes, out_sizes=out_sizes)
output_data = {'model': mod, 'test_score': score,
               'input_union': union,
               'output_union': output_union,
               'lrf': mat}
joblib.dump(output_data, model_file)
