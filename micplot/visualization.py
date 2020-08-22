import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import math
import seaborn as sns

class Consultant:
    
    def __init__(self, highlight_color='purple'):
        '''
        TODO: write docstring
        '''
        self.highlight_color = highlight_color
        self.contrasting_text_color = _return_contrasting_text_color(self.highlight_color)
        
    #TODO: method does not use self; look into this
    def recommmend_plottype(self, data):
        '''
        Determines plottype base on shape and content of data
        Based on MIcompany training

        Parameters
        ----------
        data: pandas Series or DataFrame

        Returns
        -------
        plottype: string of plottype to be used by micompanyify
        '''
        if isinstance(data.index, pd.DatetimeIndex):
            if len(data) < 10:
                plottype = 'bar_timeseries'
            else:
                plottype = 'line_timeseries'
        elif isinstance(data, pd.Series):
            if is_percentage_series(data):
                plottype = 'waterfall'
            else:
                plottype = 'bar'
        elif isinstance(data, pd.DataFrame):
            if data.apply(is_percentage_series).all():
                plottype = 'composition_comparison'
            else:
                plottype = 'scatter'
        return plottype
    
    def determine_stringformat(self, data):
        #TODO: Write docstring
        #TODO: take datatype into account
        #TODO: Shouldn't we check that the data to be plotted is numerical?
        if (isinstance(data, pd.DataFrame) and data.apply(is_percentage_series).all()) or\
            (isinstance(data, pd.Series) and is_percentage_series(data)):
            strfmt = '.1%'
        else:
            strfmt = '.2f'
        return strfmt
    
    def define_colors(self, highlight, data, plottype):
        '''
        Returns a list of colors with appropiate highlights

        Parameters
        ----------
        highlight: integer index or list of indices of rows which should be highlighted
        data: the series or dataframe for which the colors are calculated
        plottype: string of which plottype to use

        Returns
        -------
        color: list of len(data) with colors and appropriate highlights
        '''
        if isinstance(data, pd.Series) or plottype == 'scatter':
            color = ['lightgray'] * len(data)
        elif isinstance(data, pd.DataFrame):
            lendata = data.shape[1]
            color = ['lightgray'] * lendata
            if plottype == 'line_timeseries':
                cmap = matplotlib.cm.get_cmap('Greys')
                color = [cmap(x) for x in np.linspace(1/lendata, 1, lendata)]
        else:
            raise TypeError('data should be of type DataFrame or Series')

        try: #TODO: Why use error catching and not just make everything into a list?
            for h in highlight:
                if h < 0: # Count from the back
                    h = len(data) + h
                color[h] = self.highlight_color
        except TypeError:
            color[highlight] = self.highlight_color
        return color


    
    def visualize(self, data, highlight=-1, plottype=None, ascending=True, strfmt=None, **kwargs):
        '''
        Nicer visualization for basic plot, returning the axis object
        Automatically chooses plot type and nicely makes up the plot area based on learnings from
        the MIcompany training

        Parameters
        ----------
        data: must be a pandas.Series or pandas.DataFrame 
        highlight: the index or list of indices of the data point you want to highlight
        plottype: inferred from the data, but can be overridden:
            'bar' (horizontal bar), 'bar_timeseries' (vertical bars), 'line_timeseries',
            'waterfall' (builddown), 'waterfall_buildup', 'composition_comparison' or 'scatter'
        ascending: sorting direction. By default largest values are shown at the 
                    top, but False is possible, or None to leave the sorting as is
        strfmt: how to format accompanying data labels above bars, e.g. ".2f" or ".1%"
        **kwargs: will be passed to pd.Dataframe.plot()

        Returns
        -------
        :class:`matplotlib.axes.Axes` object with the plot on it

        '''
        if not isinstance(data, pd.DataFrame) and not isinstance(data, pd.Series):
            raise TypeError(f'Data is not of type Series or DataFrame, but type {type(data)}')
        #TODO: Check that data is numerical
        
        #PREPARING THE PLOT        
        data = data.squeeze() # DataFrame with single column should be treated as Series

        if plottype is None:
            plottype = self.recommmend_plottype(data)

        orient = 'h' if plottype in ['bar_timeseries', 'line_timeseries'] else 'v'

        if strfmt is None:
            strfmt = self.determine_stringformat(data)
            
        color = self.define_colors(highlight, data, plottype)

        if plottype in ['scatter', 'piechart', 'composition_comparison', 'bar_timeseries',
                        'line_timeseries'] or not isinstance(data, pd.Series):
            ascending = None

        if ascending is not None:
            data = data.sort_values(ascending=ascending)   

        # PLOTTING
        if plottype == 'bar':
            ax = data.plot(kind='barh', color=color, **kwargs)
            if isinstance(data, pd.Series):
                _plot_values_above_bar(data, orient=orient, strfmt=strfmt, ax=ax)
            else:
                for i, col in enumerate(data.columns):
                    _plot_values_above_bar(data[col], orient=orient, strfmt=strfmt,
                                          bar_nr=i, nr_bars=len(data.columns), ax=ax)

        elif plottype == 'bar_timeseries':
            ax = data.plot(kind='bar', color=color, **kwargs)
            if isinstance(data, pd.Series):
                _plot_values_above_bar(data, orient=orient, strfmt=strfmt, ax=ax)
            else:
                for i, col in enumerate(data.columns):
                    _plot_values_above_bar(data[col], orient=orient, strfmt=strfmt,
                                          bar_nr=i, nr_bars=len(data.columns), ax=ax)

        elif plottype == 'waterfall':
            ax, data, blank = self.plot_waterfall(data, color=color, **kwargs)
            _plot_values_above_bar((data+blank), data, ax=ax, strfmt=strfmt, orient=orient)
        elif plottype == 'waterfall_buildup':
            ax, data, blank = self.plot_waterfall(data, color=color, buildup=True, **kwargs)
            _plot_values_above_bar((data+blank), data, ax=ax, strfmt=strfmt, orient=orient)


        elif plottype == 'line_timeseries':
            ax = data.plot(color=color, **kwargs)

        elif plottype == 'scatter':
            x = data.iloc[:, 0]
            y = data.iloc[:, 1]
            try:
                size = data.iloc[:, 2]
            except IndexError:
                size = None
            ax = sns.scatterplot(x=x, y=y, size=size, color='grey', **kwargs)
            for i, xi, yi in zip(data.index, x, y):
                ax.annotate(i, (xi, yi))

        elif plottype == 'composition_comparison':
            ax = data.transpose().plot(kind='barh', stacked=True, color=color, **kwargs)
            data_begin = data.cumsum().shift().fillna(0)
            data_end = data.cumsum()
            location = data_begin.add(data_end).div(2)
            if not isinstance(highlight, int):
                raise TypeError('Can only highlight one line in composition comparison')

            _plot_values_above_bar(location.iloc[highlight, :], data.iloc[highlight, :],
                                  textcolor=self.contrasting_text_color, orient=orient, strfmt=strfmt, ax=ax)

        elif plottype == 'piechart':
            raise TypeError('A piechart? Are you kidding me?')
        else:
            raise NotImplementedError(f'plottype {plottype} not available, choose "bar", "waterfall", "waterfall_buildup"'
                                     ', "bar_timeseries", "line_timeseries", "scatter" or "composition_comparison"')
        
        # MAKE UP AXIS
        ax.set_frame_on(False)
        if orient == 'v' and plottype != 'scatter':
            plt.xticks([])
        elif orient == 'h' and plottype not in  ['line_timeseries', 'scatter']:
            plt.yticks([])
        return ax
    
    def plot_profile(self, data: pd.DataFrame, absolute=False, **kwargs):
        '''
        Plot a profile

        Parameters
        ----------
        data: Dataframe with two columns; the first containing the overall population and the second one the interesting part
        absolute: Whether the difference should be calculated in absolute terms or as a ratio

        Returns
        -------
        axes: 3 Axes object
        '''

        fig, axes = plt.subplots(1, 3)
        name0 = data.columns[0]
        name1 = data.columns[1]

        axes[2].set_title(f'{"Difference between" if absolute else "Ratio of"} $\it{name0}$ and $\it{name1}$')

        axes[0].set_title(name0)
        data.iloc[:, 0].pipe(self.visualize, ax=axes[0], ascending=None, plottype='bar', **kwargs)

        axes[1].set_title(name1)
        data.iloc[:, 1].fillna(0).pipe(self.visualize, ax=axes[1], ascending=None, plottype='bar', **kwargs)
        if absolute:
            outcome = data.iloc[:, 1].sub(data.iloc[:, 0], fill_value=0)
        else:
            outcome = data.iloc[:, 1].div(data.iloc[:, 0], fill_value=0)
        outcome.pipe(self.visualize, ax=axes[2], ascending=None, plottype='bar', strfmt='.1f' if absolute else '.1%', **kwargs)

        plt.axvline(0 if absolute else 1, c='k', linestyle='--')
        plt.tight_layout()

        return axes


    def plot_waterfall(self, data, color=None, buildup=False, **kwargs):
        '''
        Plot a buildup or builddown waterfall chart from data
        This function was adapted from https://pbpython.com/waterfall-chart.html

        Parameters
        ----------
        data: pd.Series to be shown as waterfall
        color: optionally give color as a list for each bar (to highlight some bars)
        buildup: False (default) for builddown, True for buildup

        Returns
        -------
        ax: Axis object
        data: the data, including a "total"-row
        blank: the size of the blank space before each bar
        '''
        if color is None:
            color = ['lightgray'] * len(data)

        blank = data.cumsum().shift(1).fillna(0)
        total = data.sum()
        data.loc['Total'] = total
        blank.loc['Total'] = 0
        color = color + ['gray']

        if buildup:
            data = data[::-1]
            blank = blank[::-1]
            color = color[::-1]

        ax = data.plot(kind='barh', stacked=True, left=blank, color=color, **kwargs)
        return ax, data, blank
    
def _plot_values_above_bar(data, display_values=None, ax=None, strfmt='.2f', orient='v',
                          textcolor='black', bar_nr=None, nr_bars=None):
    '''
    Helper function to display values above (or on) bar plot)
    
    Parameters
    ----------
    data: the series containing the position of the labels
    display_values: Optionally: the values to display if different from data
    ax: Axis object on which the labels should be plotted
    strfmt: the format-string for the labels
    orient: "v"(ertical) bars or "h"(orizontal) bars
    bar_nr: if the plot contains multiple bar charts, which of the series we want to label
    nr_bars: if the plot contains multiple bar charts, how many series are shown in total
    '''
    if (bar_nr is None) != (nr_bars is None):
        raise ValueError('Either both bar_nr and nr_bars should be None, or neither')
    
    display_values = data if display_values is None else display_values
    if ax is None:
        ax = plt.gca()
    
    if orient == 'v':
        lim_min, lim_max = ax.get_xlim()
    elif orient == 'h':
        lim_min, lim_max = ax.get_ylim()
    else:
        raise ValueError(f'orient must be "v" or "h", not {orient}')

    plotsize = lim_max - lim_min 
    for i, (v, dv) in enumerate(zip(data, display_values)):
        # offset label little bit
        offset = 0.025 * plotsize
        
        if v < 0:
            v -= offset
            
            if orient == 'v':
                ha = 'right'
            else:
                va = 'top'
        else:
            v += offset
            if orient == 'v':
                ha = 'left'
            else:
                va = 'bottom'
        
        if nr_bars is not None:
            i += -0.5 + 1/(nr_bars+2) * (bar_nr + 1)
        
        x = v if orient == 'v' else i
        y = i if orient == 'v' else v
        label = '{:{prec}}'.format(dv, prec=strfmt)
        ax.text(x, y, label, color=textcolor, va=va, ha=ha)

def _is_percentage_series(series):
    '''
    Checks whether a series contains all percentages
    
    By checking whether all values are between 0 and 1, and the sum is equal to 1
    '''
    return series.between(0, 1).all() and math.isclose(series.sum(), 1)



def _return_contrasting_text_color(colorname):
        red, green, blue, alpha = colors.to_rgba(colorname)
        if (red*0.299 + green*0.587 + blue*0.114) > 0.6:
            return 'black'
        return 'white'



