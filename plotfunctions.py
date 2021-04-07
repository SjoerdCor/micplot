# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:11:26 2021

@author: Gebruiker
"""

import pandas as pd

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
    
    def define_steps(blank: pd.Series):
        index = []
        values = []
        bar_width = 0.25
        for i, (v, v_last) in enumerate(zip(blank, blank.shift(-1))):
            index.extend([i + bar_width, i, i - bar_width])
            values.extend([v, None, v_last])
        return pd.Series(values, index=index)

    data = data.copy()
    
    blank = data.cumsum().shift(1).fillna(0)
    
    # data.loc['Total'] = total
    blank.loc["Total"] = data.loc['Total'] # This is only to get the steps right - it will later correctly be set to 0

    #The steps graphically show the levels as well as used for label placement
    step = define_steps(blank)
        
    blank.loc['Total'] = 0

    if buildup:
        data = data[::-1]
        blank = blank[::-1]
        color = color[::-1]

    ax = data.plot(kind='barh',
                   stacked=True,
                   left=blank,
                   **kwargs)
    ax.plot(step.values, step.index, 'k', linewidth=0.5)
    
    return ax

def plot_vertical_bar(data, **kwargs):
    ax = data.plot(kind='bar', **kwargs)
    return ax

def plot_line(data, **kwargs):
    ax = data.plot(kind='line', **kwargs)
    return ax