import xarray as xr
import torch
from lib.torch import column_run, ForcedStepper
import logging
import hug

logger = logging.getLogger(__name__)


@hug.cli()
@hug.local()
def main(inputs: str, forcings: str, state: str, output: str, RCE: bool =False,
         nsteps: int = 1):

    inputs = xr.open_dataset(inputs)
    forcings = xr.open_dataset(forcings)

    model = ForcedStepper.from_file(state)
    model.eval()

    model.nsteps = nsteps
    print("nsteps", nsteps)

    if RCE:
        print("Running in RCE mode (time homogeneous forcings)")
        inputs = inputs * 0 + inputs.mean('time')
        forcings = forcings * 0 + forcings.mean('time')

    progs, prec = column_run(model, inputs, forcings)
    xr.merge((progs, prec.to_dataset(name="prec")))\
      .assign(p=inputs.p)\
      .to_netcdf(output)

try:
    snakemake
except NameError:
    main.interface.cli()
else:
    main(snakemake.input.inputs,
     snakemake.input.forcings,
     snakemake.input.state,
     snakemake.output[0],
     **snakemake.params)
