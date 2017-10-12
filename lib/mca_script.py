import os
import numpy as np
from sklearn.externals import joblib

from lib.mca import MCA

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
I = np.eye(x_train.shape[1])
mat = mod.transform(I) - mod.transform(I*0)
mat = np.diag(np.sqrt(win)/scale_in) @ mat

imat = mod.inverse_transform(np.eye(mod.n_components))
imat = imat @ np.diag(scale_in/np.sqrt(win))

projection = mat @ imat

output_data ={
    'model': mod,
    'mat': mat,
    'imat': imat,
    'projection': projection,
    'transformed': (x_transformed, y_transformed),
    'explained_var': (x_explained_var, y_explained_var)
}

print("Saving output")
joblib.dump(output_data, snakemake.output[0])

