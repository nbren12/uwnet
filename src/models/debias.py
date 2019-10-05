import xarray as xr
from uwnet.debias import insert_apparent_sources, LassoDebiasedModel
import torch


o = snakemake.output
p = snakemake.params

ds = xr.open_dataset(p.data)
model = torch.load(p.model)
ds = insert_apparent_sources(ds, prognostics=p.prognostics)

mapping = [
    ('QT', 'QT', 'QQT'),
    ('SLI', 'SLI', 'QSLI'),
]

debias = LassoDebiasedModel(model, mapping).fit(ds)
torch.save(debias, o[0])

