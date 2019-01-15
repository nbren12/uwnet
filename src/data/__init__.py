from .sam import SAMRun
import xarray as xr
from pathlib import Path

_this_file = Path(__file__)

root = _this_file.parent.parent.parent

runs = {
    'micro': 'data/runs/2018-12-27-microphysics/',
    'dry': 'data/runs/2018-12-27-dry/',
    'debias': 'data/runs/model268-epoch5.debiased/',
    'unstable': 'data/runs/model265-epoch3'
}

runs = {key: SAMRun(root / val, 'control') for key, val in runs.items()}

training_data = str(root / "data/processed/training.nc")
ngaqua_climate_path = str(root / "data/processed/training.mean.nc")

ngaqua = root / "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX/"


def open_data(tag):
    """Open commonly used datasets"""
    if tag == "training":
        return xr.open_dataset(training_data).isel(step=0).drop('step')
    elif tag == 'ngaqua_2d':
        return xr.open_dataset(str(ngaqua / 'coarse' / '2d' / 'all.nc'))
    elif tag == 'training_with_src':
        from uwnet.thermo import compute_apparent_source
        ds = open_data('training')
        return ds.assign(
            Q1=compute_apparent_source(ds.SLI, 86400 * ds.FSLI),
            Q2=compute_apparent_source(ds.QT, 86400 * ds.FQT), )
    else:
        return NotImplementedError
