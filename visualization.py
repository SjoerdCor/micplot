# -*- coding: utf-8 -*-
"""
This modules contains all necessary elements to make an effective plot, primarily
through the `visualize` function, which is a wrapper around the Visualization class

By using `visualize(data)`, where data is a pandas Series or pandas DataFrame,
the user receives a Visualization object, with an Axis object as an attribute
that contains a plot of the data
"""

from itertools import cycle

import pandas as pd
import matplotlib.pyplot as plt

import defaults
import utils
import plotfunctions


class Annotator:
    """Annotates a plot."""

    def __init__(self, ax, orient, strfmt='.2f', text_alignment=None):
        """
        Initialize Annotator object.

        Parameters
        ----------
        ax : matplotlib Axes object
            the Axes that should be annotated
        orient : str
            The orientation of the graphic; determines direction of offset
        text_alignment : str, optional
            How to align the annotation. The default is None: determine from plot
        strfmt : str, optional
            Format specifier for the labels. The default is '.2f'.

        Returns
        -------
        None.

        """
        self.ax = ax
        if orient not in ['h', 'v']:
            raise ValueError(f'orient must be "v" or "h", not {orient}')
        self.orient = orient
        self.text_alignment = text_alignment
        self.strfmt = strfmt

    def _determine_offset(self):
        """
        Calculate offset in x or y distance, depending on plot orientation.

        By default is 2.5% of the plot width

        Returns
        -------
        offset : float
            The offset in plot distance

        """
        if self.orient == 'h':
            lim_min, lim_max = self.ax.get_xlim()
        else:
            lim_min, lim_max = self.ax.get_ylim()
        plotsize = lim_max - lim_min

        offset =  defaults.OFFSET_FRACTION * plotsize
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
            if self.orient == 'h':
                ha = self.text_alignment or 'right'
                va = 'center'
            else:
                va = self.text_alignment or 'top'
                ha = 'center'
        else:
            if self.orient == 'h':
                ha = self.text_alignment or 'left'
                va = 'center'
            else:
                va = self.text_alignment or 'bottom'
                ha = 'center'
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
        x = value if self.orient == 'h' else index
        y = index if self.orient == 'h' else value
        return (x, y)

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
            label = '{:{prec}}'.format(dv, prec=self.strfmt)
            self.ax.annotate(label, xy, xytext, va=va, ha=ha)
    def annotate_scatter(self, coordinates: pd.DataFrame, display_values=None):
        offset = self._determine_offset()
        
        for label, x, y in zip(display_values, coordinates.iloc[:, 0], coordinates.iloc[:, 1]):
            y2 = y + offset
            self.ax.annotate(label, (x, y), (x, y2))

            
        
    def annotate_dataframe(self, df: pd.DataFrame):
        """
        

        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        for i, colname in enumerate(df):
            index_offset = -0.5 + (i + 1) / (df.shape[1] + 2)
            self.annotate(df[colname], index_offset=index_offset)
        
class Consultant:
    """Recommend plotting choices. """

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
        if isinstance(data.index, pd.DatetimeIndex):
            if len(data) < defaults.LEN_LINEPLOT:
                plottype = 'vertical_bar'
            else:
                plottype = 'line'
        elif isinstance(data, pd.Series):
            if utils.is_percentage_series(data):
                plottype = 'waterfall'
            else:
                plottype = 'bar'
        elif isinstance(data, pd.DataFrame):
            if data.apply(utils.is_percentage_series).all():
                plottype = 'composition_comparison'
            elif data.shape[1] == 2:
                plottype = 'scatter'
            elif data.shape[1] == 3:
                plottype = 'bubble'
        return plottype

    def recommend_annotation(self, data, plottype=None):
        """
        Recommends whether to annotate a plot

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data which is plotted
        plottype : str, optional
            The type of plot. If not filled, recommends it based on recommended plot type

        Returns
        -------
        annotate : bool
            Whether to annotate

        """
        plottype = plottype or self.recommend_plottype(data)

        if (plottype in ['bar', 'waterfall', 'vertical_bar']
            or (plottype in ['scatter', 'bubble'] and len(data) <= defaults.LEN_ANNOTATE_SCATTER)):
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
        if isinstance(data.index, pd.DatetimeIndex):
            return 'index'
        elif isinstance(data, pd.DataFrame):
            return 'original'
        return 'ascending'

    def recommend_stringformat(self, data):
        '''
        Determine label precision from data type

        Parameters
        ----------
        data : pandas Dataframe or Series with data to label
        '''
        if (isinstance(data, pd.DataFrame) and data.apply(utils.is_percentage_series).all()) or\
            (isinstance(data, pd.Series) and utils.is_percentage_series(data)):
            strfmt = '.1%'
        elif (isinstance(data, pd.DataFrame) and data.apply(pd.api.types.is_integer_dtype).all()) or\
            (isinstance(data, pd.Series) and pd.api.types.is_integer_dtype(data)):
            strfmt = 'd'
        else:
            strfmt = '.2f'
        return strfmt
    
    def recommend_highlight_type(self, data, plottype):
        if isinstance(data, pd.Series) or plottype in ['scatter', 'bubble']:
            return 'row'
        return 'column'

    # def recommend_choices(self, data):
    #     choices = {}
    #     choices['plottype'] = self.recommend_plottype(data)
    #     choices['annotated'] = self.recommend_annotation(choices['plottype'], data)
    #     choices['highlight'] = self.recommend_highlight()
    #     return choices


class Visualization:
    """
    Visualizes the data and hold all choices as attributes.

    Fully customizable through its iniatilization and its attributes
    """

    plots = {'bar': {'function': plotfunctions.plot_bar,
                 'axes_with_ticks': ['y'],
                 'orient': 'h',
                 },
             'waterfall': {'function': plotfunctions.plot_waterfall,
                           'axes_with_ticks': ['y'],
                           'orient': 'h',
                           },
             'vertical_bar': {'function': plotfunctions.plot_vertical_bar,
                 'axes_with_ticks': ['x'],
                 'orient': 'v',
                 },
             'line': {'function': plotfunctions.plot_line,
                 'axes_with_ticks': ['x'],
                 'orient': 'v',
                 },
             'scatter': {'function': plotfunctions.plot_scatter,
                 'axes_with_ticks': ['x', 'y'],
                 'orient': 'v',
                 },
             'bubble': {'function': plotfunctions.plot_bubble,
                 'axes_with_ticks': ['x', 'y'],
                 'orient': 'v',
                 },

         }


    def __init__(self, data,
                 plottype=None,
                 highlight=None,
                 highlight_color=defaults.HIGHLIGHT_COLOR,
                 highlight_type=None,
                 sorting=None,
                 strfmt=None,
                 ):
        """
        Initialize the visualization.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            The data that is to be visualized
        plottype : str, optional
            The type of plot to use, must be in []. By default, this is inferred from the data(type)
        highlight : iterable, optional
            Iterable of indices of the values which should be highlighted. By default, is top value
        highlight_color : str, optional
            Color str in which to highlight some values. The default is defaults.HIGHLIGHT_COLOR.
        highlight_type : str, optional
            Whether to highlight "row" or "column"
        sorting : str, optional
            Whether and how to sort the data. By default, is determined from type data

        Raises
        ------
        TypeError
            If data is not of type pd.Series or pd.DataFrame

        """
        # TODO: make data property
        self.data = data
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise TypeError(f'Data is not of type Series or DataFrame, but type {type(data)}')
        # TODO: validate data is numeric
        self._data_to_plot = self.data.squeeze()

        fig, ax = plt.subplots()
        self.ax = ax

        self.consultant = Consultant()
        self._sorting = self.consultant.recommend_sorting(self._data_to_plot)
        self.sorting = sorting
        self._highlight = self.consultant.recommend_highlight()
        self.highlight = highlight
        self.highlight_color = highlight_color
        self.strfmt = strfmt or self.consultant.recommend_stringformat(self._data_to_plot)

        self._plottype = self.consultant.recommend_plottype(self._data_to_plot)
        self.plottype = plottype
        self._data_to_plot = self.prepare_data()
        self.sorting = sorting
        
        self.highlight_type = highlight_type or self.consultant.recommend_highlight_type(self._data_to_plot, self.plottype)
        self.annotated = self.consultant.recommend_annotation(self._data_to_plot, self.plottype)

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
            raise ValueError(f'Plottype must be one of {self.plots.keys()}, not `{new_plottype}`')
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
        # TODO: validate is iterable
        self._highlight = new_highlight

    @property
    def sorting(self):
        return self._sorting

    @sorting.setter
    def sorting(self, new_sorting):
        if new_sorting is None:
            new_sorting = self.consultant.recommend_sorting(self._data_to_plot)
        self._new_sorting = new_sorting

    def prepare_data(self):
        new_data = (self.data
                    .squeeze() # DataFrame with single column should be treated as Series
                    .pipe(utils.sort, self.sorting)
                    )
        if self.plottype == 'waterfall':
            new_data.loc['Total'] = new_data.sum()
        return new_data

    def _determine_annotation(self):
        # TODO: add user preference as an option
        self.annotated = self.consultant.recommend_annotation(self._data_to_plot, self.plottype)

    def _find_len_properties(self):
        axis_highlight_type = {'row': 0,
                               'column': 1}
        len_axis = axis_highlight_type[self.highlight_type]
        len_properties = self._data_to_plot.shape[len_axis]
        return len_properties

    def _define_colors(self):
        '''
        Return a list of colors with appropiate highlights.

        Returns
        -------
        color: list of len(data) with colors and appropriate highlights
        '''
        len_colors = self._find_len_properties()
        color = [defaults.BACKGROUND_COLOR] * len_colors

        # Last bar is total, which should not be highlighted
        if self.plottype == 'waterfall':
            color = color[:-1]

        for h in self.highlight:
            color[h] = self.highlight_color

        # Add darker shade for full bar
        if self.plottype == 'waterfall':
            color += [defaults.BENCHMARK_COLOR]
        return color

    def _define_linestyles(self):
        possible_linestyles = ['-','--','-.',':']
        linecycler_background = cycle(possible_linestyles)
        linecycler_highlight = cycle(possible_linestyles)
        linestyles = []
        
        len_axis = self._find_len_properties()
        for i in range(len_axis):
            linestyles.append(next(linecycler_background))
        for h in self.highlight:
            linestyles[h] = next(linecycler_highlight)
        return linestyles

    def annotate(self):
        """ Annotates values in self.ax"""
        if self.plottype == 'waterfall':
            blank = self._data_to_plot.cumsum().shift(1).fillna(0)
            blank.loc['Total'] = 0
            locations = self._data_to_plot + blank
        else:
            locations = self._data_to_plot

        ann = Annotator(self.ax, self._plot_properties['orient'], strfmt=self.strfmt)
    
        if isinstance(self._data_to_plot, pd.Series):
            ann.annotate(locations, self._data_to_plot)
        else:
            if self.plottype in ['scatter', 'bubble']:
                ann.annotate_scatter(self._data_to_plot.iloc[:, :2], self._data_to_plot.index)
            else:
                ann.annotate_dataframe(self._data_to_plot)
    

    def plot(self):
        """ Plot the data and show nicely."""
        plotter = self._plot_properties['function']
        color = self._define_colors()
        linestyles = self._define_linestyles()
        self.ax = plotter(self._data_to_plot, color=color, style=linestyles, ax=self.ax)

        if self.annotated:
            self.annotate()

        self.show_nicely()

    def show_nicely(self):
        """ Make the plot look better, by removing fluff."""

        self.ax.set_frame_on(False)

        # TODO: format ticks better, especially for datetimes
        if 'x' not in self._plot_properties['axes_with_ticks']:
            self.ax.set_xticks([])
        if 'y' not in self._plot_properties['axes_with_ticks']:
            self.ax.set_yticks([])
        
        # TODO: stop legend from overlapping

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
        The visualiziation with all choices as attribtes that can be modified

    """
    vis = Visualization(data, **kwargs)
    vis.plot()
    return vis

micompanyify = visualize

if __name__ == '__main__':
    import numpy as np
    visualize(pd.Series(np.random.rand(20)))
    visualize(pd.Series([0.8, 0.1, 0.1]))
    size = 6
    import numpy as np
    data = 10*np.random.rand(size, 3)
    test_data = pd.DataFrame(data, index=pd.date_range('20190101', periods=size))
    visualize(test_data)
    
    size = 12
    data = 10*np.random.rand(size, 3)
    test_data = pd.DataFrame(data, index=pd.date_range('20190101', periods=size))
    visualize(test_data)

    size = 12
    data = 10*np.random.rand(size, 5)
    test_data = pd.DataFrame(data, index=pd.date_range('20190101', periods=size))
    visualize(test_data, highlight=[-1, -2])
    
    scatter_data = pd.DataFrame(np.random.rand(30, 2))
    visualize(scatter_data)
    
    scatter_data = pd.DataFrame(np.random.rand(30, 3)) - 0.5
    vis = visualize(scatter_data)    
