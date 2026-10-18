"""Disclosed v0.2.5 display/default-state regression examples; fictional inputs."""
import numpy as np
import pandas as pd


def default_columns(rows, columns=['Close']):
    return rows[columns[0]].rolling(5).mean()


def kwdefault_columns(rows, *, columns=['Close']):
    return rows[columns[0]].rolling(5).mean()


def default_predictions(rows, predictions=np.arange(80.)):
    return predictions[:len(rows)]


def global_input(rows):
    return PREDICTIONS[:len(rows)]


PREDICTIONS = np.arange(80.)
