import pandas as pd
import xarray as xr
import numpy as np


def bootstrap(stat, data, n=1000):
    stats = []
    for i in range(n):
        sample = np.random.choice(data, data.shape[0])
        stats.append(stat(sample))

    return stats


def get_ci(x, interval=(2.5, 97.5)):
    samples = bootstrap(np.median, x)
    a, b = np.percentile(samples, interval)
    return np.median(x), (b - a) / 2


def get_cis(x):
    return x.apply(get_ci, axis=0)


def format_uncertain(x):
    if x[0] > 1000:
        return r'$\infty$'
    else:
        return f"{x[0]:.2f} ({x[1]:.2f})"


def get_error_tables(err):
    err = xr.open_dataset(err)
    err = err.assign(
        qt64=(err.qt * err.w).mean('z'), sl64=(err.sl * err.w).mean('z'))

    df = err[['qtR2', 'slR2', 'qt64', 'sl64', 'nhidden',
              'window_size']].to_dataframe()

    df_with_ci = df.groupby(['nhidden', 'window_size'])\
                   .apply(get_cis)\
                   .drop('window_size', 1)\
                   .drop('nhidden', 1)\
                   .applymap(format_uncertain)

    # rename the columns
    columns = [
        ('qtR2', (r'Apparent Source $R^2$', r'$q_T$')),
        ('slR2', (r'Apparent Source $R^2$', r'$s_L$')),
        ('qt64', (r'64 Step Error', r'$q_T$')),
        ('sl64', (r'64 Step Error', r'$s_L$')),
    ]

    cols = pd.MultiIndex.from_tuples(dict(columns).values())
    df_with_ci.columns = cols

    # get reorder the data frame to plot the vary T and then the Vary N
    # experiment
    cat = pd.concat([
        df_with_ci.loc[pd.IndexSlice[128, :], :],
        df_with_ci.loc[pd.IndexSlice[:, 10], :]
    ])

    return cat.reset_index()\
              .rename(columns={'nhidden': r'$n$', 'window_size': r'$T$'})
