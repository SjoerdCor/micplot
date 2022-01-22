# -*- coding: utf-8 -*-
"""
This modules contains all necessary elements to make an effective plot, primarily
through the `visualize` function, which is a wrapper around the Visualization class

By using `visualize(data)`, where data is a pandas Series or pandas DataFrame,
the user receives a Visualization object, with an Axis object as an attribute
that contains a plot of the data
"""

import warnings
import math
from itertools import cycle
from typing import Iterable, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

from . import defaults, utils, plotfunctions  # TODO: move defaults into yaml

# TODO: split module for classes


class Annotator:
    """Annotates a plot."""

    def __init__(self, ax, orient, strfmt=".2f", text_alignment=None, text_color=None):
        """
        Initialize Annotator object.

        Parameters
        ----------
        ax : matplotlib Axes object
            the Axes that should be annotated
        orient : str
            The orientation of the graphic; determines direction of offset
        strfmt : str, optional
            Format specifier for the labels. The default is '.2f'.
        text_alignment : str, optional
            How to align the annotation. The default is None: determine from plot
        text_color : str, optionnal
            Color in which to annotate values. The default is matplotlib default


        Returns
        -------
        None.

        """
        self.ax = ax
        if orient not in ["h", "v"]:
            raise ValueError(f'orient must be "v" or "h", not {orient}')
        self.orient = orient
        self.strfmt = strfmt
        self.text_alignment = text_alignment
        self.text_color = text_color

    def _determine_offset(self):
        """
        Calculate offset in x or y distance, depending on plot orientation.

        By default is 2.5% of the plot width

        Returns
        -------
        offset : float
            The offset in plot distance

        """
        if self.orient == "h":
            lim_min, lim_max = self.ax.get_xlim()
        else:
            lim_min, lim_max = self.ax.get_ylim()
        plotsize = lim_max - lim_min

        offset = defaults.OFFSET_FRACTION * plotsize
        return offset

    def _determine_alignments(self, value):
        """
        Determine text alignment of the annotation.

        Determines it with a minimal chance of overlap with the plot

        Parameters
        ----------
        value : float
            The value to annotate

        Returns
        -------
        ha : str
            Horizontal alignment
        va : str
            Vertical alignment

        """
        if value < 0:
            if self.orient == "h":
                ha = self.text_alignment or "right"
                va = "center"
            else:
                va = self.text_alignment or "top"
                ha = "center"
        else:
            if self.orient == "h":
                ha = self.text_alignment or "left"
                va = "center"
            else:
                va = self.text_alignment or "bottom"
                ha = "center"
        return ha, va

    def _determine_xy_from_value(self, value, index):
        """
        Transform meaningful plus index to x and y coordinate.

        Based on plot orientation

        Parameters
        ----------
        value : float
            The meaningful value
        index : float
            The index value

        Returns
        -------
        x : float
            x coordinate of the input
        y : float
            y coordinate of the input

        """
        x = value if self.orient == "h" else index
        y = index if self.orient == "h" else value
        return (x, y)

    # TODO: add line annotation

    def annotate(self, coordinates: pd.Series, display_values=None, index_offset=0):
        """
        Annotate the axis.

        Parameters
        ----------
        coordinates : pd.Series
            The location of the values
        display_values : pd.Series, optional
            The label of each coordinate. The default is to plot the coordinate values.
        index_offset : float, optional
            How much to displace bars - useful when annotating multiple bars: they become
            smaller, so we must align the numers

        Returns
        -------
        None.

        """
        if display_values is None:
            display_values = coordinates

        offset = self._determine_offset()

        for i, (v, dv) in enumerate(zip(coordinates, display_values)):
            ind = i + index_offset
            xy = self._determine_xy_from_value(v, ind)

            if v < 0:
                v -= offset
            else:
                v += offset
            xytext = self._determine_xy_from_value(v, ind)
            ha, va = self._determine_alignments(v)
            label = "{:{prec}}".format(dv, prec=self.strfmt)
            self.ax.annotate(label, xy, xytext, va=va, ha=ha, color=self.text_color)

    def annotate_scatter(self, coordinates: pd.DataFrame, display_values):
        """
        Annotate scatter plot from its coordinates.

        Always places labels above the data points


        Parameters
        ----------
        coordinates : pd.DataFrame
            First column contains x values, second column contains y values.
        display_values : Iterable
            Labels to plot above points. Must be of equal length of the coordinates

        Returns
        -------
        None.

        """
        offset = self._determine_offset()

        for label, x, y in zip(
            display_values, coordinates.iloc[:, 0], coordinates.iloc[:, 1]
        ):
            y2 = y + offset
            self.ax.annotate(label, (x, y), (x, y2))

    def annotate_dataframe(self, df: pd.DataFrame):
        """
        Annotate each series of the DataFrame.

        Corrects for the fact that with multiple columns, bars become smaller

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of which each series will be annotated.

        Returns
        -------
        None.

        """

        for i, colname in enumerate(df):
            index_offset = -0.5 + (i + 1) / (df.shape[1] + 2)
            self.annotate(df[colname], index_offset=index_offset)


class Consultant:
    """Recommend plotting choices."""

    def recommend_plottype(self, data):
        """
        Determine plottype based on shape and content of data.

        Based on MIcompany training

        Parameters
        ----------
        data: pandas Series or DataFrame

        Returns
        -------
        plottype: string of plottype to be used by micompanyify
        """
        # TODO: check whether with multiple categories vertical_bar still is best
        if isinstance(data.index, pd.DatetimeIndex):
            if len(data) < defaults.LEN_LINEPLOT:
                plottype = "vertical_bar"
            else:
                plottype = "line"
        elif isinstance(data, pd.Series):
            if utils.is_percentage_series(data):
                plottype = "waterfall"
            else:
                if len(data) < 50:  # More bars leads to very slow plotting
                    plottype = "bar"
                else:
                    plottype = "line"
        elif isinstance(data, pd.DataFrame):
            if data.apply(utils.is_percentage_series).all():
                plottype = "composition_comparison"
            elif data.shape[1] == 2:
                plottype = "scatter"
            elif data.shape[1] == 3:
                plottype = "bubble"
            else:
                return "bar"
        return plottype

    def recommend_annotation(self, data, plottype=None):
        """
        Recommends whether to annotate a plot.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data which is plotted
        plottype : str, optional
            The type of plot. If not filled, recommends it based on recommended
            plot type

        Returns
        -------
        annotate : bool
            Whether to annotate

        """
        plottype = plottype or self.recommend_plottype(data)

        if plottype in [
            "bar",
            "waterfall",
            "vertical_bar",
            "composition_comparison",
        ] or (
            plottype in ["scatter", "bubble"]
            and len(data) <= defaults.LEN_ANNOTATE_SCATTER
        ):
            return True

        return False

    def recommend_highlight(self):
        """
        Recommends which value(s) to highlight

        By default, recommends the top value

        Returns
        -------
        list
            indices of values to highlight

        """
        return [-1]

    def recommend_sorting(self, data):
        """
        Recommends whether and how to sort the data

        See `utils.sort` for the implementation of the sorting

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data

        Returns
        -------
        str
            sorting parameter

        """
        # TODO: check whether this is as wanted. Do you want to sort with a line plot? Shouldnt it be based on the plottype (e.g. never sort a bubble/scatter plot)
        if isinstance(data.index, pd.DatetimeIndex):
            return "index"
        if isinstance(data, pd.DataFrame):
            return "original"
        return "ascending"

    def recommend_stringformat(self, data):
        """
        Determine label precision from data type

        Parameters
        ----------
        data : pandas Dataframe or Series with data to label
        """
        if (
            isinstance(data, pd.DataFrame)
            and data.apply(utils.is_percentage_series).all()
        ) or (isinstance(data, pd.Series) and utils.is_percentage_series(data)):
            strfmt = ".1%"
        elif (
            isinstance(data, pd.DataFrame)
            and data.apply(pd.api.types.is_integer_dtype).all()
        ) or (isinstance(data, pd.Series) and pd.api.types.is_integer_dtype(data)):
            strfmt = "d"
        else:
            strfmt = ".2f"
        return strfmt

    def recommend_highlight_type(self, data, plottype):
        row_plottypes = ["scatter", "bubble", "composition_comparison"]
        if isinstance(data, pd.Series) or plottype in row_plottypes:
            return "row"
        return "column"

    def recommend_reference_line(self, plottype):
        if plottype == "bar":
            return ["mean"]
        return []


class Visualization:
    """
    Visualize the data and hold all choices as attributes.

    Fully customizable through its iniatilization and its attributes
    """

    plots = {
        "bar": {
            "function": plotfunctions.plot_bar,
            "axes_with_ticks": ["y"],
            "orient": "h",
        },
        "waterfall": {
            "function": plotfunctions.plot_waterfall,
            "axes_with_ticks": ["y"],
            "orient": "h",
        },
        "vertical_bar": {
            "function": plotfunctions.plot_vertical_bar,
            "axes_with_ticks": ["x"],
            "orient": "v",
        },
        "line": {
            "function": plotfunctions.plot_line,
            "axes_with_ticks": ["x", "y"],
            "orient": "v",
        },
        "scatter": {
            "function": plotfunctions.plot_scatter,
            "axes_with_ticks": ["x", "y"],
            "orient": "v",
        },
        "bubble": {
            "function": plotfunctions.plot_bubble,
            "axes_with_ticks": ["x", "y"],
            "orient": "v",
        },
        "pie": {"function": plotfunctions.plot_pie},
        "composition_comparison": {
            "function": plotfunctions.plot_composition_comparison,
            "axes_with_ticks": ["y"],
            "orient": "h",
        },
    }

    def __init__(
        self,
        data,
        plottype: str = None,
        highlight: Union[Iterable[int], int] = None,
        highlight_color: str = defaults.HIGHLIGHT_COLOR,
        highlight_type: str = None,
        sorting: str = None,
        annotated: bool = None,
        strfmt: str = None,
        reference_lines: Iterable[Union[str, int]] = None,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """
        Initialize the visualization.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data that is to be visualized
        plottype : str, optional
            The type of plot to use. By default, this is inferred from the data(type).
            Must be one of:
                - 'bar'
                - 'vertical_bar'
                - 'waterfall'
                - 'line'
                - 'scatter'
                - 'bubble'
                - 'pie'
                - 'composition_comparison'
        highlight : iterable, optional
            Iterable of indices of the values which should be highlighted. By default, is top value
        highlight_color : str, optional
            Color str in which to highlight some values. The default is defaults.HIGHLIGHT_COLOR.
            Other allowed specification types are documented here: https://matplotlib.org/stable/tutorials/colors/colors.html
        highlight_type : str, optional
            Whether to highlight "row" or "column". By default, this is determined from the data
        sorting : str, optional
            Whether and how to sort the data. By default, is determined from the data (type)
            Must be one of:
            - 'original': do not sort the data
            - 'index': sort the index of the data ascending
            - 'ascending': sort data ascending
            - 'descending': sort data descending
        annotated : bool, optional
            Whether values should also be displayed in text. By default, this is
            inferred from the data
        strfmt : str, optional
            The format string, how to annotate the data. By default, this is inferred
            from the data type
        reference_lines : iterable, optional
            The numerical values or aggregation to perform at which reference lines are shown.
            By default, in a bar plot, the mean is shown.
        ax : plt.Axes, optional
            The Axes object on which to plot. If not given, a new Axes is created with the **kwargs
        kwargs
            Passed to plt.subplots(), e.g. figsize

        Raises
        ------
        TypeError
            If data is not of type pd.Series or pd.DataFrame

        """
        # TODO: make data property
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise TypeError(
                f"Data is not of type Series or DataFrame, but type {type(data)}"
            )
        # TODO: validate data is numeric
        self.data = data
        self._data_to_plot = (
            self.data.copy()
        )  # We must do this quickly because there is a bit of a circular dependency: plottype depends on data, but data depends on plottype
        # TODO: look into this

        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        self.ax = ax

        self.consultant = Consultant()
        self._sorting = self.consultant.recommend_sorting(self._data_to_plot)
        self.sorting = sorting
        self.highlight_color = highlight_color
        self.strfmt = strfmt or self.consultant.recommend_stringformat(
            self._data_to_plot
        )

        self._plottype = self.consultant.recommend_plottype(self._data_to_plot)
        self.plottype = plottype
        self._data_to_plot = self.prepare_data()
        self.sorting = sorting

        self.highlight_type = (
            highlight_type
            or self.consultant.recommend_highlight_type(
                self._data_to_plot, self.plottype
            )
        )
        # Highlights can only be determined after the highlight type
        self._highlight = self.consultant.recommend_highlight()
        self.highlight = highlight  # TODO why does this happen in two lines again?

        # TODO: allow for secondary highlight colors
        if self.plottype == "line" and len(self.highlight) > 3:
            msg = """You have more than 3 highlighted lines in a line plot, since
            there are no more line styles, you will not be able to distinguish all
            highlighted lines. Consider highlighting less lines, or using another plottype
                    """
            warnings.warn(msg)
        if self.plottype == "vertical_bar" and len(self.highlight) > 1:
            msg = """You are highlighting more than one category in the vertical bar chart.
            You will not be able to distinguish those. Consider highlight only one category,
            or switching to a line plot.
            """
            warnings.warn(msg)

        if annotated is None:
            annotated = self.consultant.recommend_annotation(
                self._data_to_plot, self.plottype
            )
        self.annotated = annotated

        # reference_lines as an empty list is a valid input but returns False
        if reference_lines is not None:
            self.reference_lines = reference_lines
        else:
            self.reference_lines = self.consultant.recommend_reference_line(
                self.plottype
            )

    @property
    def plottype(self):
        return self._plottype

    @plottype.setter
    def plottype(self, new_plottype):
        """
        Changes plottype after validation

        If new_plottype is None, the recommended plottype is set
        Raises ValueError if not an allowed plottype
        """
        if new_plottype is None:
            new_plottype = self.consultant.recommend_plottype(self._data_to_plot)
        new_plottype = new_plottype.lower()
        if new_plottype not in self.plots.keys():
            raise ValueError(
                f"Plottype must be one of {self.plots.keys()}, not `{new_plottype}`"
            )
        self._plottype = new_plottype
        self._determine_annotation()
        self._plot_properties = self.plots[self.plottype]
        # self.plot()

    @property
    def highlight(self):
        return self._highlight

    @highlight.setter
    def highlight(self, new_highlight):
        """
        Changes which data points to highlight

        Parameters
        ----------
        new_highlight : Iterable

        """
        if new_highlight is None:
            new_highlight = self.consultant.recommend_highlight()
        if isinstance(new_highlight, int):
            new_highlight = [new_highlight]

        len_axis = self._find_len_properties()
        highlight_def = []

        # All positive list is easier to work with when iterating in other methods
        for h in new_highlight:
            if h >= 0:
                highlight_def.append(h)
            else:
                highlight_def.append(len_axis + h)
        self._highlight = highlight_def

    @property
    def sorting(self):
        return self._sorting

    @sorting.setter
    def sorting(self, new_sorting):
        """
        Changes how to sort the data.

        Parameters
        ----------
        new_sorting : str, optional
            Must be in ['original', 'index', 'ascending', 'descending']
        """
        if new_sorting is None:
            new_sorting = self.consultant.recommend_sorting(self._data_to_plot)
        self._sorting = new_sorting

    @property
    def reference_lines(self):
        return self._reference_lines

    @reference_lines.setter
    def reference_lines(self, new_reference_lines):
        """
        Changes how to sort the data.

        Parameters
        ----------
        new_sorting : str, optional
            Must be in ['original', 'index', 'ascending', 'descending']
        """
        if new_reference_lines is None:
            new_reference_lines = self.consultant.recommend_reference_line(
                self.plottype
            )
        self._reference_lines = new_reference_lines

    def prepare_data(self):
        """
        Make data uniform and plotworthy.
        """
        new_data = self.data.copy()  # Never modify the original data

        # We dont want to squeeze a single number Series to a non-pandas type
        # That would crash the Visualization which works with pandas types
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.squeeze()

        new_data = new_data.pipe(utils.sort, self.sorting)
        if self.plottype == "waterfall":
            # TODO: this is not allowed if the index is Categorical <- make sure it's not or add category?
            new_data.loc["Total"] = new_data.sum()
        return new_data

    def _determine_annotation(self):
        # TODO: add user preference as an option
        self.annotated = self.consultant.recommend_annotation(
            self._data_to_plot, self.plottype
        )

    def _find_len_properties(self):
        axis_highlight_type = {"row": 0, "column": 1}
        len_axis = axis_highlight_type[self.highlight_type]
        len_properties = self._data_to_plot.shape[len_axis]
        return len_properties

    def _define_colors(self):
        """
        Return a list of colors with appropiate highlights.

        Returns
        -------
        color: list of len(data) with colors and appropriate highlights
        """
        len_axis = self._find_len_properties()

        def define_line_colors(len_axis, highlight, highlight_color):
            n_linestyles = 3
            n_non_highlighted = len_axis - len(self.highlight)
            if n_non_highlighted == 0:
                return [highlight_color] * len_axis

            n_different_colors = math.ceil(n_non_highlighted / n_linestyles)
            # TODO: the cmap should be some sort of config or config related
            non_highlight_colorlist = [
                matplotlib.cm.get_cmap("Greys")(x)
                for x in np.linspace(1 / n_different_colors, 1, n_different_colors)
            ]
            non_highlight_colorlist = np.repeat(
                non_highlight_colorlist, n_linestyles, axis=0
            )
            non_highlight_colorlist = cycle(non_highlight_colorlist)
            color = []
            for i in range(len_axis):
                if i in highlight:
                    color.append(self.highlight_color)
                else:
                    color.append(next(non_highlight_colorlist))
            return color

        if self.plottype == "line":
            color = define_line_colors(len_axis, self.highlight, self.highlight_color)

        else:
            color = [defaults.BACKGROUND_COLOR] * len_axis
            for h in self.highlight:
                color[h] = self.highlight_color

        # Last bar is total, which should not be highlighted
        if self.plottype == "waterfall":
            color = color[:-1]

        # Add darker shade for full bar
        if self.plottype == "waterfall":
            color += [defaults.BENCHMARK_COLOR]
        return color

    def _define_linestyles(self):
        possible_linestyles = ["-", "--", ":"]
        linecycler_background = cycle(possible_linestyles)
        linecycler_highlight = cycle(possible_linestyles)
        linestyles = []

        len_axis = self._find_len_properties()
        for _ in range(len_axis):
            linestyles.append(next(linecycler_background))
        for h in self.highlight:
            linestyles[h] = next(linecycler_highlight)
        return linestyles

    def annotate(self):  # TODO: break function into smaller pieces
        """ Annotates values in self.ax."""
        if self.plottype == "waterfall":
            blank = self._data_to_plot.cumsum().shift(1).fillna(0)
            blank.loc["Total"] = 0
            locations = self._data_to_plot + blank
            display_values = self._data_to_plot

        elif self.plottype == "composition_comparison":
            data_begin = self._data_to_plot.cumsum().shift().fillna(0)
            data_end = self._data_to_plot.cumsum()
            if len(self.highlight) > 1:
                raise TypeError("Can only highlight one line in composition comparison")

            locations = data_begin.add(data_end).div(2)
            display_values = self._data_to_plot
            if isinstance(locations, pd.DataFrame):
                locations = locations.iloc[self.highlight[0]]
                display_values = self._data_to_plot.iloc[self.highlight[0]]
            elif isinstance(
                locations, pd.Series
            ):  # We must keep it as a Series, not squeeze it to int/float
                locations = locations.iloc[[self.highlight[0]]]
                display_values = self._data_to_plot.iloc[[self.highlight[0]]]
        else:
            locations = self._data_to_plot
            display_values = self._data_to_plot

        if self.plottype == "composition_comparison":
            # With a composition comparison, annotating is done _in_ the plot
            # intead of just outside it
            text_color = utils.contrasting_text_color(self.highlight_color)
            text_alignment = "center"
        else:
            text_color = None
            text_alignment = None

        ann = Annotator(
            self.ax,
            self._plot_properties["orient"],
            strfmt=self.strfmt,
            text_color=text_color,
            text_alignment=text_alignment,
        )

        if isinstance(display_values, pd.Series):
            ann.annotate(locations, display_values)
        elif isinstance(display_values, pd.DataFrame):
            if self.plottype in ["scatter", "bubble"]:
                ann.annotate_scatter(
                    self._data_to_plot.iloc[:, :2], self._data_to_plot.index
                )
            else:
                ann.annotate_dataframe(self._data_to_plot)
        else:
            raise NotImplementedError(
                f"Cannot display annotation for type {type(display_values)}"
            )

    def add_reference_line(self, value="mean", text: str = None, c: str = "k"):
        """
        Add dashed reference line to visualizaiton

        Parameters
        ----------
        value : number or aggregation method on pd.Series/pd.DataFrame
            The default is 'mean'.
        text : str, optional
            description of what the vlaue represents. The default is None.
        c : str, optional
            color of the reference value. The default is 'k'.
        """
        if isinstance(value, (float, int)):
            ref_val = value
        else:
            ref_val = self._data_to_plot.agg(value)

        if self._plot_properties["orient"] == "h":
            line = self.ax.axvline
            annot_xy = (ref_val, 0)
            annot_xy_text = (ref_val * 1.02, 0.1)

        else:
            line = self.ax.axhline
            annot_xy = (0, ref_val)
            annot_xy_text = (0.1, ref_val * 1.02)
        line(ref_val, c=c, ls="--")

        # Cannot alway use strfmt of visualization:
        # Average of ints can be a float (whcih should not be formatted as int)
        if self.strfmt == "d" and isinstance(ref_val, float):
            prec = ".1f"
        else:
            prec = self.strfmt
        annotation = "{:{prec}}".format(ref_val, prec=prec)
        if text is not None:
            annotation += f", {text}"
        elif isinstance(value, str):
            annotation += f", {value}"
        self.ax.annotate(annotation, annot_xy, annot_xy_text, c=c)
        return self

    def plot(self):
        """ Plot the data and show nicely."""
        plotter = self._plot_properties["function"]
        color = self._define_colors()
        linestyles = self._define_linestyles()
        self.ax = plotter(self._data_to_plot, color=color, style=linestyles, ax=self.ax)

        for v in self.reference_lines:
            self.add_reference_line(v)

        if self.annotated:
            self.annotate()

        self.show_nicely()

    def show_nicely(self):
        """ Make the plot look better, by removing fluff."""

        self.ax.set_frame_on(False)

        # TODO: format ticks better, for datetimes and integers
        # TODO: format ticks based on strfmt

        if "x" not in self._plot_properties["axes_with_ticks"]:
            self.ax.set_xticks([])
        if "y" not in self._plot_properties["axes_with_ticks"]:
            self.ax.set_yticks([])

        # Plot contains legend
        if (
            isinstance(self._data_to_plot, pd.DataFrame)
            # Legend is fixed inside plotting function
            and self.plottype not in ["scatter", "bubble"]
        ):
            if self.highlight_type == "row":
                title = self._data_to_plot.index.name
            else:
                title = self._data_to_plot.columns.name
            utils.move_legend_outside_plot(self.ax, title=title)

        if isinstance(self._data_to_plot, pd.Series):
            name = self._data_to_plot.name or self._data_to_plot.index.name
            if self._plot_properties["orient"] == "v" or self.plottype == "waterfall":
                self.ax.set_ylabel(name)
            else:
                self.ax.set_xlabel(name)


def visualize(data, **kwargs):
    """
    Visualize data and return the visualization containing all attributes

    See Visualization for full information

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        The data to visualize
    **kwargs
        See Visualization documentation

    Returns
    -------
    vis : `cls::Visualization`
        The visualiziation with all choices as attributes that can be modified

    """
    vis = Visualization(data, **kwargs)
    vis.plot()
    return vis


micompanyify = visualize
