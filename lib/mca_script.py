import os
import numpy as np
from sklearn.externals import joblib

from lib.mca import MCA
from lib.util import mat_to_xarray

n_components = snakemake.params.n_components
data = joblib.load(snakemake.input[0])

x_train, y_train = data['train']
win, wout = data['w']
scale_in, scale_out = data['scale']

# scale and weight variables
x_train = x_train*np.sqrt(win)/scale_in
y_train = y_train*np.sqrt(wout)/scale_out

# fit model
mod = MCA(n_components=n_components)
print(f"Fitting {mod}")
mod.fit(x_train, y_train)

# output transforms
print(f"Computing transformation")
x_transformed, y_transformed = mod.transform(x_train, y_train)

print("Computing variance explained statistics")
x_explained_var = mod.x_explained_variance_(x_train)
y_explained_var = mod.y_explained_variance_(y_train)
print(f"R2 x: {x_explained_var}, R2 y{y_explained_var}")

# compute matrix and inverse
print("Computing transformation matrix")
mat = np.diag(np.sqrt(win)/scale_in) @ mod.x_components_
imat = mod.x_components_.T @ np.diag(scale_in/np.sqrt(win))

projection = mat @ imat

# input and output modes for plotting
# these modes should have the original physical units
# which means that the weights should be normalized in some
# reasonable way
print("Computing input and output modes")
input_modes = imat
output_modes = mod.y_components_.T @ np.diag(scale_out/np.sqrt(wout))
# add dimension information
input_modes = mat_to_xarray(input_modes, {1: x_train['features']})
output_modes = mat_to_xarray(output_modes, {1: y_train['features']})

output_data ={
    'model': mod,
    'mat': mat,
    'imat': imat,
    'projection': projection,
    'transformed': (x_transformed, y_transformed),
    'explained_var': (x_explained_var, y_explained_var),
    'modes': (input_modes, output_modes)
}

print("Saving output")
joblib.dump(output_data, snakemake.output[0])

