import os
import numpy as np
import xarray as xr
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lib.sklearn import Select, Stacker, Weighter, WeightedNormalizer
from toolz.curried import map, pipe

from lib.mca import MCA

mem = joblib.Memory("/tmp/mycache")

# snakemake input and outputs
data3d = snakemake.input.data3d
data2d = snakemake.input.data2d
weight = os.path.abspath(snakemake.input.weight)
model_file = snakemake.output[0]

data3d, data2d = [[os.path.abspath(x) for x in files]
                  for files in [data3d, data2d]]


def load_data():
    """Load and merge 2d and 3d datasets
    """
    D = xr.open_mfdataset(data3d)
    D2 = xr.open_mfdataset(data2d)
    w = xr.open_dataarray(weight)

    D = D.merge(D2, join='inner')

    return D, w


D, w = load_data()
w /= w.sum()
D['Q1c'] = D.Q1 - D.QRAD
d = D.sel(time=slice(20, None))


def pipeline_var(name, w):
    return make_pipeline(
        Select(name),
        WeightedNormalizer(w),
        Weighter(np.sqrt(w)),
        Stacker(['z']))


def pipeline_2d_var(name):
    return make_pipeline(
        Select(name),
        Stacker(),
        StandardScaler())


union = make_union(
    pipeline_var('QT', w),
    pipeline_var('SL', w),
    pipeline_2d_var('LHF'),
    pipeline_2d_var('SHF'))

output_union = make_union(
    pipeline_var('Q1c', w),
    pipeline_var('Q2', w)
)



x = union.fit_transform(d)
y = output_union.fit_transform(d)

mca = make_pipeline(union, MCA(y_transformer=output_union))
mca.fit(d, d)
joblib.dump(mca, model_file)

