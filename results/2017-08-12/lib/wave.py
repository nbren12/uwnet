"""Methods for computing wave-frame averages of simulations
"""
from gnl.xarray import phaseshift_regular_grid

class Wave(object):

    def __init__(self, wave):
        self._wave = wave


    @property
    def speed(self):

        x,y,xx,yy = self._wave['coord']
        return (xx-x)/(yy-y)
    @property
    def _slices(self):
        x,y,xx,yy = self._wave['coord']

        if x > xx: x,xx= xx, x
        if y > yy: y,yy= yy, y

        return dict(x=slice(x,xx), time=slice(y,yy))

    def shift(self, data, speed=None):
        if speed is None:
            speed= self.speed

        data = data.sel(time=self._slices['time'])
        return phaseshift_regular_grid(data, speed)


    def average(self, u):
        u_shifted = self.shift(u)
        avg = u_shifted.mean('time')

        up=phaseshift_regular_grid(u_shifted-avg, -self.speed)

        return avg, up


    def ishift(self, data):
        return phaseshift_regular_grid(data, -self.speed)

    @property
    def pts(self):
        x,y,xx,yy = self._wave['coord']

        return (x,xx), (y,yy)


    def plot_xt(self, A, **kw):
        import matplotlib.pyplot as plt
        assert set(A.dims) == {'x', 'time'}
        A.sel(**self._slices).plot.contourf(levels=21, **kw)
        plt.plot(*self.pts, 'k--')
        plt.title(f"Speed = {self.speed:.2f}")
