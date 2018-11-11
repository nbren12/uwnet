import os
from os.path import join
import json

import click

from sam.case import InitialConditionCase, get_ngqaua_ic, default_parameters

NGAQUA_ROOT = "/Users/noah/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"



@click.command()
@click.argument('path')
@click.option(
    '-nn',
    '--neural-network',
    type=click.Path(),
    help='use the neural network in this pickled model file.')
@click.option(
    '-mom-nn',
    '--momentum-neural-network',
    type=click.Path(),
    help='use the neural network in this pickled model file.')
@click.option('-n', '--ngaqua-root', type=click.Path(), default=NGAQUA_ROOT)
@click.option('-t', type=int, default=0)
@click.option('-r', '--run', is_flag=True)
@click.option('-d', '--docker-image', type=str, default='nbren12/uwnet')
@click.option('-p', '--parameters', type=click.Path(), default=None)
def main(path,
         neural_network,
         momentum_neural_network,
         ngaqua_root,
         t,
         run,
         docker_image,
         model_run_path='model.pkl',
         momentum_model_run_path='momentum.pkl',
         parameters=None):
    """Create SAM case directory for an NGAqua initial value problem and optionally
    run the model with docker.

    """
    if parameters:
        parameters = json.load(open(parameters))
    else:
        parameters = default_parameters()

    ic = get_ngqaua_ic(ngaqua_root, t)

    case = InitialConditionCase(
        path=path, ic=ic, sam_src="/opt/sam", docker_image=docker_image,
        prm=parameters)

    # dt = 120.0
    # day = 86400
    # hour = 3600
    # minute = 60
    # time_stop = 2 * day
    #
    # output_interval_stat = 30 * minute
    # output_interval_2d = 1 * hour
    # output_interval_3d = 1 * hour
    #
    # case.prm['parameters']['dt'] = dt
    # case.prm['parameters']['nstop'] = int(time_stop // dt)
    # case.prm['parameters']['nsave3d'] = int(output_interval_3d // dt)
    # case.prm['parameters']['nsave2d'] = int(output_interval_2d // dt)
    # case.prm['parameters']['nstat'] = int(output_interval_2d // dt)
    # case.prm['parameters']['nstat'] = int(output_interval_stat // dt)
    # case.prm['parameters']['nstatfrq'] = 1  # int(output_interval_stat // dt)
    # case.prm['parameters']['nprint'] = int(output_interval_stat // dt)
    # case.prm['parameters']['ncycle_max'] = 20

    # configure neural network run
    if neural_network:
        case.prm['python']['dopython'] = False

        # setup the neural network
        case.prm['python'].update(
            dict(
                dopython=True,
                usepython=True,
                function_name='call_neural_network',
                module_name='uwnet.sam_interface'))

        case.mkdir()

        print(f"Copying neural networks to model directory")
        case.add(neural_network, model_run_path)
        case.env.update(dict(UWNET_MODEL=model_run_path))

        if momentum_neural_network:
            case.add(momentum_neural_network, momentum_model_run_path)
            case.env.update(dict(UWNET_MOMENTUM_MODEL=momentum_model_run_path))

    case.save()

    if run:
        case.run_docker()


if __name__ == '__main__':
    main()
