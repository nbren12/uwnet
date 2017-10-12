import os

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

data = joblib.load(snakemake.input[0])

x_train, y_train = data['train']
x_test, y_test = data['test']

scale_in, scale_out = data['scale']


mod = make_pipeline(VarianceThreshold(.001), LinearRegression())
mod.fit(x_train/scale_in, y_train)
score = mod.score(x_test, y_test)

# compute matrix
I = np.eye(x_train.shape[1])
mat = mod.predict(I) - mod.predict(I * 0)
mat = np.diag(1/scale_in) @ mat

output_data = {
    'condition_number': np.linalg.cond(x_train),
    'test_score': score,
    'model': mod,
    'mat': mat,
    'features': {'in': x_test.indexes['features'],
                 'out': y_test.indexes['features']}}

joblib.dump(output_data, snakemake.output[0])
