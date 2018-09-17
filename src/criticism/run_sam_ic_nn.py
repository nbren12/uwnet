import os
from os.path import join

import click

from sam.case import InitialConditionCase, get_ngqaua_ic

NGAQUA_ROOT = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"


@click.command()
@click.argument('model', type=click.Path())
@click.option('-n', '--ngaqua-root', type=click.Path(), default=NGAQUA_ROOT)
@click.option('-t', type=int, default=0)
def main(model, ngaqua_root, t):

    ic = get_ngqaua_ic(ngaqua_root, t)

    case = InitialConditionCase(
        path='test',
        ic=ic,
        sam_src="ext/sam/",
        docker_image='nbren12/uwnet')

    case.prm['parameters']['dodamping'] = True
    case.prm['parameters']['khyp'] = 1e16

    dt = 100.0
    day = 86400
    hour = 3600
    minute = 60
    time_stop = 1 * hour

    output_interval_stat = 300.0
    output_interval_2d = 300.0
    output_interval_3d = 300.0

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

    # setup the neural network
    case.prm['python'].update(
        dict(
            dopython=True,
            usepython=True,
            npython=1,
            function_name='call_neural_network',
            module_name='uwnet.sam_interface'))

    # configure needed environmental variables
    case.mkdir()
    model_run_path = 'model.pkl'
    case.add(model, model_run_path)
    case.env.update(dict(
        UWNET_MODEL=model_run_path
    ))

    case.save()

    case.run_docker()


if __name__ == '__main__':
    main()
