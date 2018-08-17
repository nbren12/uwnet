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
 Noah D. Brenowitz nbren12 <nbren12@uw.edu> - http://www.noahbrenowitz.com
----------------------------------------------------------------------------------------
*/

params.trainingDB = "$baseDir/runs.json"
params.config = "$baseDir/examples/all.yaml"
params.numTimePoints = 640
params.NGAquaDir = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"
params.forcingMethod = 'SAM' // Available options are FD and SAM, ORIG
params.train = false // Train the model (default: false)
params.numEpochs = 1

i = Channel.from(0..params.numTimePoints-1)

/*
 Process the NGAqua data
 */

time_step_ch = Channel.from(0..params.numTimePoints - 1)


process processOneTimeStep {
    input:
        val t from time_step_ch

    output:
        file 'rec.*.nc' into single_time_steps_ch

    """
    process_ngaqua.py -n $params.NGAquaDir $t
    f=\$(ls *.nc)
    ncks -A --mk_rec_dmn time \$f rec.\$f
    rm -f \$f
    """
}


process concatAllFiles {
    publishDir 'data'
    input:
        file x from single_time_steps_ch.collect()
    output:
        file 'training_data.nc' into training_data_ch
    """
    echo $x | sort -n | ncrcat -o training_data.nc
    """
}


/*
 Train the Neural network
 */

training_data_ch.into {t1; t2}

process trainModel {
  cache ! params.train
  publishDir "data/models/${params.forcingMethod}/"
  input:
  file x from t1

  output:
        file '*.pkl' into single_column_mode_ch, sam_run_ch


  """
  python -m uwnet.check_data $x && \
  python -m uwnet.train  -n $params.numEpochs -lr .001 -s 5 -l 20 -db $params.trainingDB \
         $params.config $x
  """
}


process testTrainModel {
  cache false
  input:
  file x from t2

  when:
      false

  """
  python -m uwnet.train  --test -lr .001 -s 5 -l 20 -db $params.trainingDB \
         $params.config $x
  """
}

/*

 Criticism of the trained neural network

 1. Perform single column model simulation
 */

process runSingleColumnModel {
  input:
  file model from single_column_mode_ch
  file data from training_data_ch

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
   publishDir "data/plots/${params.forcingMethod}"
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



      for key in ['qt', 'sl', 'QN', 'QP']:
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


/*
 Run the neural network with SAM
 */
process runSAMNeuralNetwork {
    cache false
    publishDir 'data/samNN/'
    validExitStatus 0,9
    afterScript "rm -rf   RUNDATA RESTART "
    input:
        file(x) from sam_run_ch.flatten().last()

    output:
        set file('NG1/data.pkl'), file('*.pkl' ) into check_sam_dbg_ch
        file('output.txt')
    """
    run_sam_nn_docker.sh $x $baseDir/assets/NG1 > output.txt
    """
}

process checkSAMNN {
    publishDir 'data/samNN/checks'
    input:
        set file(model), file(x) from check_sam_dbg_ch
    output:
        file 'sam_nn.txt'
    // when:
    //     false

    """
    for file in $x
    do
        echo "Checking Water budget for \$file" >> sam_nn.txt
        check_sam_nn_debugging.py \$file $model >> sam_nn.txt
        echo
    done
    """

}

