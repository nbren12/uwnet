from .sam import SAMRun
from .ngaqua import NGAqua
import xarray as xr
from pathlib import Path

_this_file = Path(__file__)

root = _this_file.parent.parent.parent

run_paths = {
    'micro': 'data/runs/2018-12-27-microphysics/',
    'debias': 'data/runs/samnn/nn/NNLower/epoch4/',
    'khyp1e15': 'data/runs/samnn_khyp1e15/nn/NNLower/epoch4/',
#     'no_debias': 'data/runs/samnn/nn/NNManuscript/epoch5/',
    'unstable': 'data/runs/samnn/nn/NNAll/epoch5/',
}

runs = {key: SAMRun(root / val, 'control') for key, val in run_paths.items()}

training_data = str(root / "data/processed/training/noBlur.nc")
ngaqua_climate_path = str(root / "data/processed/training.mean.nc")

ngaqua = root / "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX/"


def open_data(tag):
    """Open commonly used datasets"""
    if tag == "training":
        return xr.open_dataset(training_data)
    elif tag == 'ngaqua_2d':
        return xr.open_dataset(str(ngaqua / 'coarse' / '2d' / 'all.nc'))\
            .sortby('time')
    elif tag == 'training_with_src':
        return open_data('training').pipe(assign_apparent_sources)
    elif tag == 'pressure':
        return open_data('training').p.isel(time=0).drop('time')
    elif tag == 'nudge':
        return runs['nudge']
    else:
        raise NotImplementedError


def open_ngaqua():
    return NGAqua(ngaqua)


def assign_apparent_sources(ds):
    from uwnet.thermo import compute_apparent_source
    return ds.assign(
        Q1=compute_apparent_source(ds.SLI, 86400 * ds.FSLI),
        Q2=compute_apparent_source(ds.QT, 86400 * ds.FQT))
