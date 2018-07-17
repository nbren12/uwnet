import numpy as np

from .interface import step_model


def test_step_model():

    def step(x, dt):
        return {'qt': x['qt'], 'sl': x['sl']}, None

    nz = 10
    shape_3d = (nz, 1, 1)
    shape_2d = (1, 1, 1)

    dt = 0.0
    kwargs = dict(
        dt=dt,
        layer_mass=np.ones((nz, )),
        qt=np.random.rand(*shape_3d),
        sl=np.random.rand(*shape_3d),
        FQT=np.random.rand(*shape_3d),
        FSL=np.random.rand(*shape_3d),
        U=np.random.rand(*shape_3d),
        V=np.random.rand(*shape_3d),
        SST=np.random.rand(*shape_2d),
        SOLIN=np.random.rand(*shape_2d), )

    out = step_model(step, **kwargs)

    # The step function above does not alter the state
    # so the step_model function should not either

    for key in out:
        if key in kwargs:
            np.testing.assert_allclose(out[key], kwargs[key])
