import os

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

data = joblib.load(snakemake.input[0])

x_train, y_train = data['train']
x_test, y_test = data['test']

scale_in, scale_out = data['scale']
weight_in, weight_out = data['w']


mod = make_pipeline(
    VarianceThreshold(.001),
    StandardScaler(),
    LinearRegression())

mod.fit(x_train, y_train)

# compute score
y_pred = mod.predict(x_test)
score = r2_score(y_test*np.sqrt(weight_out),
                 y_pred*np.sqrt(weight_out),
                 multioutput="uniform_average")
print("Score", score)

# compute matrix
I = np.eye(x_train.shape[1])
mat = mod.predict(I) - mod.predict(I * 0)
# mat = np.diag(1/scale_in) @ mat

output_data = {
    'condition_number': np.linalg.cond(x_train),
    'test_score': score,
    'model': mod,
    'mat': mat,
    'features': {'in': x_test.indexes['features'],
                 'out': y_test.indexes['features']}}

joblib.dump(output_data, snakemake.output[0])
