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

params.numTimePoints = 640
params.NGAquaDir = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"

/*
 Process the NGAqua data
 */

time_step_ch = Channel.from(0..params.numTimePoints - 1)
stat_file_ch = Channel.fromPath(params.NGAquaDir + "/stat.nc")
sfc_file_ch = Channel.fromPath(params.NGAquaDir + "/coarse/2d/all.nc")


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
    input:
        file x from single_time_steps_ch.collect()
    output:
        file 'training_data.nc' into concated_file_ch
    """
    echo $x | tr ' ' '\\n' | sort -n | ncrcat -o training_data.nc
    ncatted -a units,FQT,c,c,'g/kg/s' \
            -a units,FSLI,c,c,'K/s' \
            -a units,FU,c,c,'m/s^2' \
            -a units,FV,c,c,'m/s^2' \
            -a units,x,c,c,'m' \
            -a units,y,c,c,'m' \
             training_data.nc
    """
}


process addOtherNeededVariables {
    input:
        file x from concated_file_ch
        file stat from stat_file_ch
        file d2d from sfc_file_ch
    output:
        set file('aux.nc'), file(x) into combine_vars_ch

    """
    #!/usr/bin/env python
    import xarray as xr
    from uwnet.thermo import layer_mass

    # compute layer_mass
    stat = xr.open_dataset("$stat")
    rho = stat.RHO.isel(time=0).drop('time')
    w = layer_mass(rho)

    # get 2D variables
    d2 = xr.open_dataset("$d2d")
    ds = xr.open_dataset("$x")
    d2 = d2.sel(time=ds.time)

    out = xr.Dataset({
       'RADTOA': d2.LWNT - d2.SWNT,
       'RADSFC': d2.LWNS - d2.SWNS,
       'layer_mass': w
    })


    for key in 'Prec SHF LHF SOLIN SST'.split():
        out[key] = d2[key]

    # append these variables
    out.to_netcdf("aux.nc")

    """
}

process combineVariables {
    publishDir 'data'
    input:
        set file(x), file(y) from combine_vars_ch
    output:
        file y into training_data_ch
    """
    ncks -A $x $y
    """
}

process subsetData {
    publishDir 'data'
    input:
    file x from training_data_ch

    output:
    file 'subset.nc'
    """
    ncks -d time,0,200 -d x,0,16 $x subset.nc
    """
}
