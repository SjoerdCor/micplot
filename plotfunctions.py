# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:11:26 2021

@author: Gebruiker
"""

def plot_bar(data, **kwargs):
    ax = data.plot(kind='barh', **kwargs)
    return ax