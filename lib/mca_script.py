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

def compute_mat(mca, union, w,
                splits=[34, 68, 69],
                n=70):
    """Compute matrix for mca

    Warning this code is extremely brittle and should not be reused.
    """
    # compute LRF
    I = np.eye(n)
    lrf = mca.named_steps['mca'].transform(I)
    lrf -= mca.named_steps['mca'].transform(I*0)

    scales = {
        'qt': union.transformer_list[0][1].named_steps['weightednormalizer'].x_scale_,
        'sl': union.transformer_list[1][1].named_steps['weightednormalizer'].x_scale_,
        'lhf': union.transformer_list[2][1].named_steps['standardscaler'].scale_,
        'shf': union.transformer_list[3][1].named_steps['standardscaler'].scale_,
    }
    scales = {key: float(val) for key, val in scales.items()}
    w = w.data
    W = np.sqrt(np.diag(np.hstack((w, w, 1, 1))))

    lrf = W @ lrf
    lrf_dict = {key: val/scales[key] for  key, val in
                zip(['qt', 'sl', 'lhf', 'shf'],
                    np.split(lrf, splits))}

    return lrf_dict


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


mat = compute_mat(mca, union, w)
joblib.dump({'model': mca, 'mat': mat},
            model_file)
