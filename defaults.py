import plotfunctions

plots = {'bar': {'function': plotfunctions.plot_bar,
                 'axes_with_ticks': ['y'],
                 'orient': 'h',
                 'len_axis': 0,
                 },
         }

HIGHLIGHT_COLOR = 'purple'

# ANNOTATION
OFFSET_FRACTION = 0.025  # pct of plot to offset annotations
LEN_ANNOTATE_SCATTER = 50  # Nr of datapoints to still recommend annotating a scatter plot

# PLOTTYPE
LEN_LINEPLOT = 10 # nr of datapoints to recommend lineplot over bar plot
