```
def model(input):
    """Compute appararent sources of humidity (QT) and temperature (SLI)

    Parameters
    ----------
    input : dict
        dict of tensors. Each tensor should have the shape (* , z, y, x). * can
        be batches, time, etc. Units of QT (g/kg), units of SLI (K).

    Returns
    -------
    sources : dict
        dict of tensor output. The itme QT will be the apparent source of
        humidity in units (g/kg/day). SLI will be apparent heating in (K/day).
        The shapes will be the same as the corresponding input variables
        (*,z,y,x).

    Notes
    -----
    should be able use like this~:
        input_next_step = model(input) * (dt_days) + input
    """
```