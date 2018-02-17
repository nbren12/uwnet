from xnoah.xcalc import centderiv
from xnoah.sam.regrid import destagger


def material_derivative(u, v, w, f):

    df = vertical_advection(w, f) \
         + horizontal_advection(u, v, f)

    try:
        df.attrs['units'] = f.units.strip() + '/s'
    except AttributeError:
        pass

    return df


def vertical_advection(w, f):
    wc = destagger(w, 'z', mode='clip')
    return wc * centderiv(f, dim='z', boundary='nearest')


def horizontal_advection(u, v, f):
    return u * centderiv(f, dim='x', boundary='periodic')\
        + v * centderiv(f, dim='y', boundary='nearest')
