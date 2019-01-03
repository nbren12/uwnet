from .sam import SAMRun
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

ngaqua = root / "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX/"
