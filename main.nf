#!/usr/bin/env nextflow
// -*- mode: groovy; -*-
/*
========================================================================================
                         sam-uwnet
 ========================================================================================
 sam-uwnet Analysis Pipeline. Started 2018-08-02.
 #### Homepage / Documentation
 https://github.com/sam-uwnet
 #### Authors
 Noah D. Brenowitz nbren12 <nbren12@uw.edu> - noahbrenowitz.com>
----------------------------------------------------------------------------------------
*/

params.trainingDB = "$baseDir/runs.json"
params.config = "$baseDir/examples/all.yaml"
params.numTimePoints = 639
params.NGAquaDir = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"

i = Channel.from(0..params.numTimePoints)

process makeTrainingData {
    publishDir "data"

    output:
    file '*.zarr'  into train_data_ch

    """
    python -m uwnet.data  $params.config training_data.zarr
    """
}

/*
 Compute forcings using SAM
 */

process runSAMWithInitialConditions {
    afterScript "rm -rf OUT*  RUNDATA NGAqua RESTART _*.nc"
    input:
    val i

    output:
    set val(i), file('out.nc') into sam_short_run_ch
    stdout info logger_sam

    """
    sam.py -d \$PWD --physics dry -t $i $params.NGAquaDir out.nc
    """

}

process computeForcing {
    input:
        set val(i), file(x) from sam_short_run_ch

    output:
        file "*.nc" into forcing_ch

    """
    #!/usr/bin/env python
    import xarray as xr

    f = xr.open_dataset("$x")
    fqt = (f.QV[1] - f.QV[0]).assign_coords(time=f.time[0]).expand_dims('time')
    fqt.attrs['units'] = 'g/kg/day'

    fsl = (f.TABS[1] - f.TABS[0]).assign_coords(time=f.time[0]).expand_dims('time')
    fsl.attrs['units'] = 'K/day'

    xr.Dataset({'FSL': fsl, 'FQT': fqt})\
      .to_netcdf("${sprintf("%020d",i)}.nc", unlimited_dims=['time'])

    """

}


process combineSingleTimeStepForcings {
    publishDir 'data/sam_forcing'
    input:
        file(x) from forcing_ch.collect()
    output:
        file 'forcing.nc'
    """
    ncrcat *.nc forcing.nc
    """
}

/*
 Train the Neural network
 */

process trainModel {
  input:
  file data from train_data_ch

  output:
  file '*.pkl' into trained_model_ch


  """
  python -m uwnet.train -n 5 -lr .001 -s 5 -l 10 -db $params.trainingDB $params.config $data 
  """
}

/*

 Criticism of the trained neural network

 1. Perform single column model simulation
 */

process runSingleColumnModel {
  input:
  file model from trained_model_ch
  file data from train_data_ch

  output:
  file 'cols.nc' into single_column_ch


  """
  python -m uwnet.columns ${model.last()} $data cols.nc
  """
}

/*
 Visualize the trained model results
 */

process plotPrecipTropics {
   publishDir 'data/plots/'
   input:
   file sim from single_column_ch

   output:
   file '*.png'

  """
  #!/usr/bin/env python
  import xarray as xr
  import matplotlib.pyplot as plt
  g = xr.open_dataset("$sim").swap_dims({'z': 'p'})

  for j in [32, 20, 10]:
      plt.figure()
      g['Prec'][:,j,0].plot()
      g['PrecOBS'][:,j,0].plot()
      plt.savefig(f"prec_{j}.png")



      for key in ['qt', 'sl']:
          plt.figure()
          qt = g[key][:,:,j, 0]
          qt.plot.contourf(x='time')
          plt.gca().invert_yaxis()
          plt.savefig(f"{key}_{j}.png")

          plt.figure()
          qt = g[key][:,5,j, 0]
          plt.plot(g.time, qt)
          plt.savefig(f"{key}_{j}_z5.png")

  """

}
