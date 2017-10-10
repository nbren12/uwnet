import os
import numpy as np
import xarray as xr
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union
from lib.sklearn import Select, Stacker, Weighter

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

output_data = {'model': mod, 'test_score': score,
               'input_union': union,
               'output_union': output_union}

joblib.dump(output_data, model_file)
