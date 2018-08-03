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

process makeTrainingData {
    publishDir "data"

    output:
    file '*.zarr'  into train_data_ch

    """
    python -m uwnet.data  $params.config training_data.zarr
    """
}

process trainModel {
  input:
  file data from train_data_ch

  output:
  file '*.pkl' into trained_model_ch


  """
  python -m uwnet.train -n 1 -lr .001 -s 5 -l 10 $params.config $data 
  """
}

process runSingleColumnModel {
  input:
  file model from trained_model_ch.last()
  file data from train_data_ch

  output:
  file 'cols.nc' into single_column_ch


  """
  python -m uwnet.columns $model $data cols.nc
  """
}

// Visualization processes


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

  for j in [32, 10]:
      plt.figure()
      g['Prec'][:,j,0].plot()
      g['PrecOBS'][:,j,0].plot()
      plt.savefig(f"prec_{j}.png")



  for key in ['qt', 'sl']:
      plt.figure()
      qt = g[key][:,:,32, 0]
      qt.plot.contourf(x='time')
      plt.gca().invert_yaxis()
      plt.savefig(f"{key}.png")

  for key in ['qt', 'sl']:
      plt.figure()
      qt = g[key][:,5,32, 0]
      plt.plot(g.time, qt)
      plt.savefig(f"{key}_z5.png")

  """

}