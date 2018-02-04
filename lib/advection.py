from xnoah.xcalc import centderiv
from xnoah.sam.coarsen import destagger


def material_derivative(u, v, w, f):

    wc = destagger(w, 'z', mode='clip')

    # compute total derivative
    df = u * centderiv(f, dim='x', boundary='periodic')\
        + v * centderiv(f, dim='y', boundary='nearest')\
        + wc * centderiv(f, dim='z', boundary='nearest')

    try:
        df.attrs['units'] = f.units.strip() + '/s'
    except AttributeError:
        pass

    return df
