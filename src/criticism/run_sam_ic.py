from sam.case import InitialConditionCase, get_ngqaua_ic

ic = get_ngqaua_ic(
    "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX", 0)

case = InitialConditionCase(
    ic=ic,
    path="data/runs/dosfc_dodamp_dosgs_sponge600-10800",
    sam_src="ext/sam/",
    exe='/opt/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM',
    docker_image='nbren12/uwnet')

case.prm['parameters']['dodamping'] = True
case.prm['parameters']['khyp'] = 1e16

dt = 100.0
day = 86400
hour = 3600
minute = 60
time_stop = 2 * day

output_interval_stat = 30 * minute
output_interval_2d = 1 * hour
output_interval_3d = 3 * hour

case.prm['parameters']['dt'] = dt
case.prm['parameters']['nstop'] = int(time_stop // dt)
case.prm['parameters']['nsave3d'] = int(output_interval_3d // dt)
case.prm['parameters']['nsave2d'] = int(output_interval_2d // dt)
case.prm['parameters']['nstat'] = int(output_interval_2d // dt)
case.prm['parameters']['nstat'] = int(output_interval_stat // dt)
case.prm['parameters']['nstatfrq'] = 1  # int(output_interval_stat // dt)
case.prm['parameters']['nprint'] = int(output_interval_stat // dt)

case.prm['parameters']['dosgs'] = True
case.prm['parameters']['dosurface'] = True
case.prm['python']['dopython'] = False

case.run_docker()
