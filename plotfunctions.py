# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:11:26 2021

@author: Gebruiker
"""

def plot_bar(data, **kwargs):
    ax = data.plot(kind='barh', **kwargs)
    return ax

def plot_waterfall(data, buildup=False, **kwargs):
    '''
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
    '''
    data = data.copy()
    
    blank = data.cumsum().shift(1).fillna(0)
    
    # data.loc['Total'] = total
    blank.loc["Total"] = data.loc['Total'] # This is only to get the steps right - it will later correctly be set to 0

    #The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = None # Do not connect bars with themselves
    
    # TODO: start connecting lines at top of bar, end at bottm, by changing index
    
    blank.loc['Total'] = 0

    if buildup:
        data = data[::-1]
        blank = blank[::-1]
        color = color[::-1]
    display(step)
    display(data)
    display(blank + data)
    ax = data.plot(kind='barh',
                   stacked=True,
                   left=blank,
                   **kwargs)
    ax.plot(step.values, step.index, 'k', linewidth=1)
    
    return ax#, data, blank
