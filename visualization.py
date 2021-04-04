# -*- coding: utf-8 -*-
"""
This modules contains all necessary elements to make an effective plot, primarily
through the `visualize` function, which is a wrapper around the Visualization class

By using `visualize(data)`, where data is a pandas Series or pandas DataFrame,
the user receives a Visualization object, with an Axis object as an attribute
that contains a plot of the data
"""
import pandas as pd

import defaults
import utils


class Annotator:
    """Annotates a plot."""
    
    def __init__(self, ax, orient, text_alignment=None, strfmt='.2f'):
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
            raise ValueError(f'orient must be "v" or "h", not {self.orient}')
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
        if self.orient == 'v':
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
        va : TYPE
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

    def annotate(self, coordinates: pd.Series, display_values=None):
        """
        Annotate the axis.

        Parameters
        ----------
        coordinates : pd.Series
            The location of the values
        display_values : pd.Series, optional
            The label of each coordinate. The default is to plot the coordinate values.

        Returns
        -------
        None.

        """
        if display_values is None:
            display_values = coordinates

        offset = self._determine_offset()

        for i, (v, dv) in enumerate(zip(coordinates, display_values)):
            xy = self._determine_xy_from_value(v, i)
            
            if v < 0:
                v -= offset
            else:
                v += offset
            xytext = self._determine_xy_from_value(v, i)
            ha, va = self._determine_alignments(v)
            label = '{:{prec}}'.format(dv, prec=self.strfmt)
            self.ax.annotate(label, xy, xytext, va=va, ha=ha)


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
                plottype = 'bar_timeseries'
            else:
                plottype = 'line_timeseries'
        elif isinstance(data, pd.Series):
            if utils.is_percentage_series(data):
                plottype = 'waterfall'
            else:
                plottype = 'bar'
        elif isinstance(data, pd.DataFrame):
            if data.apply(utils.is_percentage_series).all():
                plottype = 'composition_comparison'
            else:
                plottype = 'scatter'
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
        
        if (plottype in ['bar']
            or (plottype == 'scatter' and len(data) <= defaults.LEN_ANNOTATE_SCATTER)):
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

    # def recommend_choices(self, data):
    #     # TODO: determine strfmt
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
    def __init__(self, data, plottype=None,
                 highlight=None, highlight_color=defaults.HIGHLIGHT_COLOR,
                 sorting=None):
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

        self.ax = None

        self.consultant = Consultant()
        self._sorting = self.consultant.recommend_sorting(self._data_to_plot)
        self.sorting = sorting
        self._highlight = self.consultant.recommend_highlight()
        self.highlight = highlight
        self.highlight_color = highlight_color

        self._data_to_plot = self.prepare_data()
        self.sorting = sorting

        self._plottype = self.consultant.recommend_plottype(self._data_to_plot)
        self.plottype = plottype
        self.annotated = self.consultant.recommend_annotation(self._data_to_plot, self._plottype)

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
        if new_plottype not in defaults.plots.keys():
            raise ValueError(f'Plottype must be one of {defaults.plots.keys()}, not `{new_plottype}`')
        self._plottype = new_plottype
        self._determine_annotation()
        self._plot_properties = defaults.plots[self.plottype]
        self.plot()

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
        return new_data

    def _determine_annotation(self):
        # TODO: add user preference as an option
        self.annotated = self.consultant.recommend_annotation(self._data_to_plot, self.plottype)

    def _define_colors(self):
        '''
        Returns a list of colors with appropiate highlights

        Parameters
        ----------
        highlight: list of indices of rows which should be highlighted
        data: the series or dataframe for which the colors are calculated
        plottype: string of which plottype to use

        Returns
        -------
        color: list of len(data) with colors and appropriate highlights
        '''

        len_colors = self._data_to_plot.shape[self._plot_properties['len_axis']]
        color = ['lightgray'] * len_colors
        
        for h in self.highlight:
            color[h] = self.highlight_color
        return color

    def annotate(self):
        Annotator(self.ax, self._plot_properties['orient']).annotate(self._data_to_plot)

    def plot(self):
        plotter = self._plot_properties['function']
        color = self._define_colors()

        self.ax = plotter(self._data_to_plot, color=color)
        if self.annotated:
            self.annotate()

        self.show_nicely()

    def show_nicely(self):

        self.ax.set_frame_on(False)

        if 'x' not in self._plot_properties['axes_with_ticks']:
            self.ax.set_xticks([])
        if 'y' not in self._plot_properties['axes_with_ticks']:
            self.ax.set_yticks([])


def visualize(data, **kwargs):
    vis = Visualization(data, **kwargs)
    vis.plot()
    return vis

micompanyify = visualize

if __name__ == '__main__':
    visualize(pd.Series([1, 3, 2]))
