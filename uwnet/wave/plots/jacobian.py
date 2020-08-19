from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def plot(args):
    """Plot jacobian

    Parameters
    ----------
    args : tuple
        (jac, p) tuple
    """
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(5, 4))
    jac, p = args

    k = 0

    abc = 'abcd'

    progs = list(jac)

    for i, outkey in enumerate(progs):
        for j, inkey in enumerate(progs):
            ax = axs[i, j]
            val = jac[outkey][inkey] * 86400
            # select the colorbar norm
            # vmax = get_vmax(val)
#             do_sym_log = vmax > 10

#             if do_sym_log:
#                 linthresh = 1.0
#                 norm = SymLogNorm(
#                     linthresh=linthresh, linscale=20, vmin=-vmax, vmax=vmax)
#             else:
#                 norm = Normalize(vmin=-vmax, vmax=vmax)
#             if inkey == 'QT':
#                 norm = Normalize(vmin=-1.5, vmax=1.5)
#             else:
#                 norm =  Normalize(vmin=-1.0, vmax=1.0)

            im = ax.pcolormesh(p, p, val, cmap='RdBu_r')
            # ax.contourf(
            #     p, p, val, levels=[-100,-10,10, 100] , extend='both', colors='none',
            #     hatches=['xxxx', '...', None, '...', 'xxxx'],)

            letter = abc[k]
            ax.set_title(f"{letter}) d{outkey}/dt from {inkey}", loc='left')
            ax.set_xlabel(f"Pressure ({inkey})")
            ax.set_ylabel(f"Pressure (d{outkey}/dt)")
            ax.invert_yaxis()
            ax.invert_xaxis()

            cb = plt.colorbar(im, ax=ax, pad=-.04)

#             if do_sym_log:
#                 locator = matplotlib.ticker.SymmetricalLogLocator(
#                     linthresh=linthresh, base=10)
#                 locator.set_params(numticks=5)
#                 cb.locator = locator
#                 cb.update_ticks()

            k += 1

    return axs