import os
import numpy as np
from sklearn.externals import joblib

from lib.mca import MCA


data = joblib.load(snakemake.input[0])

x_train, y_train = data['train']
win, wout = data['w']
scale_in, scale_out = data['scale']

# fit model
mod = MCA(n_components=4)
mod.fit(x_train*np.sqrt(win)/scale_in, y_train * np.sqrt(wout)/scale_out)

# compute matrix and inverse
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
    'projection': projection
}

joblib.dump(output_data, snakemake.output[0])

