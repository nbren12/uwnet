import os

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline


from lib.models import WeightedOutput, prepvar, weights_to_np

data = joblib.load(snakemake.input[0])

x_train, y_train = data['train']
x_test, y_test = data['test']


mod = make_pipeline(VarianceThreshold(.001), LinearRegression())
mod.fit(x_train, y_train)
score = mod.score(x_test, y_test)

output_data = {
    'condition_number': np.linalg.cond(x_train),
    'test_score': score,
    'model': mod,
    'features': {'in': x_test.indexes['features'],
                 'out': y_test.indexes['features']}}

joblib.dump(output_data, snakemake.output[0])
