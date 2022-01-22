# -*- coding: utf-8 -*-
"""Utility functions to help plotting."""
import re
import math
from typing import Union, Callable

import pandas as pd
import matplotlib.colors


def is_percentage_series(series):
    """
    Checks whether a series contains all percentages

    By checking whether all values are between 0 and 1, and the sum is equal to 1
    """
    return series.between(0, 1).all() and math.isclose(series.sum(), 1)


def contrasting_text_color(colorname):
    """
    Calculates whether text on top of this color should be in black or white

    Taken from https://stackoverflow.com/questions/3942878
    /how-to-decide-font-color-in-white-or-black-depending-on-background-color
    """
    red, green, blue, alpha = matplotlib.colors.to_rgba(colorname)
    if (red * 0.299 + green * 0.587 + blue * 0.114) > 0.6:
        return "black"
    return "white"


def apply_function_to_series_or_all_columns(
    data: Union[pd.Series, pd.DataFrame], function: Callable
):
    """
    Applies a function to the series if data is pd.Series, or to each column if data is pd.DataFrame
    """
    if isinstance(data, pd.Series):
        return function(data)
    elif isinstance(data, pd.DataFrame):
        return data.apply(function).all()
    else:
        raise TypeError(f"Data should be pd.Series or pd.DataFrame not {type(data)!r}")


def sort(data, sorting):
    """
    Sort data based on sorting parameter.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Data to be sorted
    sorting : str
        How to sort data, must be one of ['original', 'index', 'ascending', 'descending']

    """
    # TODO: validata data is of type DataFrame or Series
    if sorting == "original":
        return data
    elif sorting == "index":
        return data.sort_index()
    elif sorting == "ascending":
        return data.sort_values()
    elif sorting == "descending":
        return data.sort_values(ascendng=False)
    raise NotImplementedError(f"Unknown sorting type `{sorting}`")


def extract_number(string: str):
    """
    Extract first float from text.

    Parameters
    ----------
    string : str
        String that contains a number.

    Returns
    -------
    number : float
        The extracted float

    """
    found = re.search("\d+\.\d+", string)
    if not found:
        found = re.search("\.\d+", string)
    if not found:
        found = re.search("\d+", string)
    number = float(found.group())
    return number


def move_legend_outside_plot(ax, **kwargs):
    """
    Draw legend outside of plot area.

    Plot area is decreased by 20% to make room for the legend

    Parameters
    ----------
    ax : The axis for which the legend must be draw

    kwargs are passed to the legend
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kwargs)


class SizeScaler:
    """ Transform numbers to numbers appropriate for scatter plot marker sizes."""

    def __init__(self, factor=8):
        """
        Set up the scaler.

        Parameters
        ----------
        factor : float, optional
            Defines how much difference there will be in sizes based on difference in value

        Returns
        -------
        None.

        """
        self.factor = factor

        self.addition = None
        self.mean = None
        self.std = None

    def fit(self, data):
        """Learn transformation from data value to marker size."""

        self.mean = data.mean()
        self.std = data.std()

        min_val = ((data - data.mean()) / data.std() * self.factor).min()

        self.addition = -1 * min_val + 7

    def _check_is_fitted(self):
        if any(value is None for value in [self.mean, self.std, self.addition]):
            raise ValueError("Scaler not fitted yet")

    def transform(self, data):
        """Transform data points to marker sizes."""
        self._check_is_fitted()
        return (data - self.mean) / self.std * self.factor + self.addition

    def fit_transform(self, data):
        """First fit, then transform data."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """Transform marker size to data value."""
        self._check_is_fitted()
        return (data - self.addition) / self.factor * self.std + self.mean
