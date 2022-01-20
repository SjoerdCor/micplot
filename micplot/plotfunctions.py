# -*- coding: utf-8 -*-
"""Separate functions for each plottype of visualization."""

import pandas as pd

from . import utils


def plot_bar(data, **kwargs):
    """ Plot bar chart."""
    ax = data.plot(kind="barh", **kwargs)
    return ax


def plot_waterfall(data, buildup=False, **kwargs):
    """
    Plot a buildup or builddown waterfall chart from data.

    This function was adapted from https://pbpython.com/waterfall-chart.html

    Parameters
    ----------
    data: pd.Series to be shown as waterfall
    buildup: False (default) for builddown, True for buildup

    Returns
    -------
    ax: Axis object
    data: the data, including a "total"-row
    blank: the size of the blank space before each bar
    """

    def define_steps(blank: pd.Series):
        """
        Steps graphically show the transitions for the levels

        Parameters
        ----------
        blank : pd.Series
            Values where the data start

        Returns
        -------
        values from which steps can be plotted

        """
        index = []
        values = []
        bar_width = 0.25
        for i, (v, v_last) in enumerate(zip(blank, blank.shift(-1))):
            index.extend([i + bar_width, i, i - bar_width])
            # None assures bars are not connected to themselves
            values.extend([v, None, v_last])
        return pd.Series(values, index=index)

    data = data.copy()

    blank = data.cumsum().shift(1).fillna(0)

    # This is only to get the steps right - it will later correctly be set to 0
    # TODO: this is not allowed if the index is Categorical <- make sure it's not or add category?
    blank.loc["Total"] = data.loc["Total"]
    step = define_steps(blank)

    blank.loc["Total"] = 0

    if buildup:
        data = data[::-1]
        blank = blank[::-1]
        color = color[::-1]

    ax = data.plot(kind="barh", stacked=True, left=blank, **kwargs)
    ax.plot(step.values, step.index, "k", linewidth=0.5)

    return ax


def plot_vertical_bar(data, **kwargs):
    """ Plot vertical bar chart, useful for timeseries."""
    ax = data.plot(kind="bar", **kwargs)
    return ax


def plot_line(data, **kwargs):
    """ Plot line chart, useful for line chart."""
    ax = data.plot(kind="line", **kwargs)
    return ax


def plot_scatter(data: pd.DataFrame, **kwargs):
    """ Plot scatter chart."""
    x = data.columns[0]
    y = data.columns[1]

    ax = data.plot(kind="scatter", x=x, y=y, **kwargs)
    return ax


def plot_bubble(data: pd.DataFrame, **kwargs):
    """ Plot bubble chart: scatter chart with different sizes per point."""

    def calculate_legend_values(ax, scaler, strfmt=".2f"):
        handles, labels = ax.collections[0].legend_elements(prop="sizes", alpha=0.6)
        positions = pd.Series([utils.extract_number(label) for label in labels])
        labels = [
            "{:{prec}}".format(x, prec=strfmt)
            for x in scaler.inverse_transform(positions)
        ]
        return handles, labels

    # TODO: add strfmt
    x = data.columns[0]
    y = data.columns[1]
    scaler = utils.SizeScaler(4)
    size = scaler.fit_transform(data.iloc[:, 2]).clip(1)
    ax = data.plot(kind="scatter", x=x, y=y, s=size, **kwargs)

    handles, labels = calculate_legend_values(ax, scaler)
    utils.move_legend_outside_plot(
        ax, handles=handles, labels=labels, title=data.columns[2]
    )
    return ax


def plot_pie(data, **kwargs):
    """ Plot pie chart - small joke."""
    raise TypeError("A pie chart? Are you kidding me?")


def plot_composition_comparison(data, **kwargs):
    """ Plot comparison of distributions as stacked bar charts."""
    ax = data.transpose().plot(kind="barh", stacked=True, **kwargs)
    return ax
