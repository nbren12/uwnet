"""Fit a LTSM model to the two state markov chain time series
"""
import pandas as pd

def _prepare_data():
    df = pd.read_csv("./time_series.csv")
