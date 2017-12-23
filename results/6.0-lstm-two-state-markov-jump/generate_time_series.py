"""Generate time series of the two state growth/decay process
Saves the resulting time series to a csv file
"time_series.csv"

"""
import numpy as np
import pandas as pd


def forcing(h, u, t1=1.0, t2=5.0):
    return - (1 - h) * (u-1)/t1 + h * u/t2


def switching_rate(h):
    return {0: 3.0, 1: 15.0}[h]


def generate(n, dt=.001):

    h = 0
    u = 1.0
    t = 0.0
    next_jump_time = -1

    ts = [t]
    hs = [h]
    us = [u]

    for i in range(n):
        # perform the deterministic step
        u = u + dt * forcing(h, u)
        t = t + dt

        # determing the hidden markov chain state
        z = np.random.rand(1)
        next_jump_time = -np.log(z)/switching_rate(h)
        if next_jump_time < dt:
            h = 1-h

        hs.append(h)
        us.append(u)
        ts.append(t)


    return pd.DataFrame({'t': ts, 'u': us, 'h': hs}).set_index('t')


def main():
    df = generate(10000, .01)
    df.to_csv("time_series.csv")

if __name__ == '__main__':
    main()
