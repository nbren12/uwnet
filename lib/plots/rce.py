"""Plots of RCE for NN, Truth and CAM
"""
import matplotlib.pyplot as plt
import xarray as xr
import lib.cam as lc


def mean(x):
    return x.isel(time=slice(-400, None)).mean('time')


def load_data(root_dir):

    truth = xr.open_dataset(f"{root_dir}/data/processed/inputs.nc")\
              .isel(x=10, y=8).pipe(mean)
    rce_cam = xr.open_dataset(f"{root_dir}/data/processed/rce/10-8/cam.nc")\
                .squeeze().pipe(mean)
    rce_nn = xr.open_dataset(f"{root_dir}/data/output/model.VaryT-20/3.rce.nc")\
               .isel(x=10, y=8).pipe(mean)

    stat_file = f"{root_dir}/data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"
    p = xr.open_dataset(stat_file).p

    # interpolate onto sam vertical grid
    rce_cam_interp = lc.to_sam_z(rce_cam, p)
    # compute sl for cam
    sl = rce_cam_interp['T'] + 9.81/1004 * rce_cam_interp.z
    qt = rce_cam_interp['Q'] * 1000
    rce_cam_interp = rce_cam_interp.assign(sl=sl, qt=qt)

    return truth, rce_nn, rce_cam_interp, p


def _plot(data):
    truth, rce_nn, rce_cam_interp, p = data

    # setup axes
    fig, (axT, axQ) = plt.subplots(1, 2, figsize=(3, 2), sharey=True)

    # plot data
    axT.plot(truth['sl'], p, label='Mean')
    axT.plot(rce_nn['sl'], rce_cam_interp.p)
    axT.plot(rce_cam_interp['sl'], rce_cam_interp.p)

    axQ.plot(truth['qt'], p, label='Mean')
    axQ.plot(rce_nn['qt'], rce_cam_interp.p, label='NN')
    axQ.plot(rce_cam_interp['qt'], rce_cam_interp.p, label='CAM')

    # legends
    axQ.legend()

    # labels
    axT.set_ylabel('p (hPa)')
    axT.set_xlabel('K')
    axQ.set_xlabel('g/kg')

    # titles
    axT.set_title("A) $s$", loc="left")
    axQ.set_title("B) $q_v$", loc="left")

    # remove spines
    for ax in [axQ, axT]:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

    # invert axis
    axT.invert_yaxis()


def plot(root_dir=""):
    data = load_data(root_dir)
    cycler = plt.cycler('color', ['k', 'b', 'g'])
    with plt.rc_context({'axes.prop_cycle': cycler}):
        _plot(data)
