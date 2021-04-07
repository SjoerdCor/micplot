# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:46:53 2021

@author: Gebruiker
"""
import re
import math
import matplotlib.colors


def is_percentage_series(series):
    '''
    Checks whether a series contains all percentages

    By checking whether all values are between 0 and 1, and the sum is equal to 1
    '''
    return series.between(0, 1).all() and math.isclose(series.sum(), 1)


def contrasting_text_color(colorname):
    '''
    Calculates whether text on top of this color should be in black or white
    
    # TODO: Add source Taken from ... 
    '''
    red, green, blue, alpha = matplotlib.colors.to_rgba(colorname)
    if (red*0.299 + green*0.587 + blue*0.114) > 0.6:
        return 'black'
    return 'white'


def sort(data, sorting):
    # TODO: validata data is of type DataFrame or Series
    if sorting == 'original':
        return data
    elif sorting == 'index':
        return data.sort_index()
    elif sorting == 'ascending':
        return data.sort_values()
    elif  sorting == 'descending':
        return data.sort_values(ascendng=False)
    raise NotImplementedError(f'Unknown sorting type `{sorting}`')

      
def extract_number(string: str):
    found = re.search('\d+\.\d+', string)
    if not found:
        found = re.search('\.\d+', string)
    if not found:
        found = re.search('\d+', string)
    return float(found.group())
