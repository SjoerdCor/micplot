# Installation
`micplot` can be installed by forking this repository and running `pip install micplot` in the appropriate folder. The only requirements are `pandas` and `matplotlib`.

# More effective visualization in one line of code
Pandas is  an extremely popular python package for data manipulation, and for good reason: it has a host of possibilities. However, it's out-of-the-box plotting options usually result in hard to interpret plots. This is unfortunate, because good visualization leads to better discussion with and more insights from subject matter experts, which is sorely needed for useful data analytics



```python
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import micplot
```


```python
df = pd.read_csv(os.path.join('..', 'data', 'titanic.csv'))
```


```python
data = df.groupby('Embarked')['Fare'].mean()
data.plot() 
plt.show()
```


    
![PNG](readme/output_3_0.PNG)
    


Indeed, the plot is difficult to interpret. A line plot is a poor choice for this type of data, there are meaningless ticks, and no axis label.

Therefore, the `micplot` package was developed, with three advantages:
  1. It automatically makes choices that make the plot much easier to interpret
  1. It makes the up the plot area nicely, by removing fluff, so it is easer to read.
  1. It is fully customizable if something is not to your wishes
  
## Creating focus
In plotting, it is important to make clear what the point of the plot is. `micplot` does this in two ways:
1. From the data, it infers a focus point, by sorting and highlighting data, and  
1. It makes the plot clearer, by annotating when necessary and removing fluff, such as unnecessary ticks and the frame.




```python
vis = micplot.visualize(data)
plt.show()
```


    
![PNG](readme/output_5_0.PNG)
    


## The plot is still fully customizable
If the plot is not fully to your liking, the `Visualization` object that is returned contains all choices as attributes, including the axis, which can still be altered. In the example below, we alter the plottype and the bars which are highlighted. 


```python
vis = micplot.visualize(data, plottype='vertical_bar', highlight=[0, 1])
vis.ax.set_ylabel('Mean ticket price')
plt.show()
```


    
![PNG](readme/output_7_0.PNG)
    


Other options that can be altered are in the documentation:


```python
?micplot.Visualization
```


    [1;31mInit signature:[0m
    [0mmicplot[0m[1;33m.[0m[0mVisualization[0m[1;33m([0m[1;33m
    [0m    [0mdata[0m[1;33m,[0m[1;33m
    [0m    [0mplottype[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mhighlight[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mhighlight_color[0m[1;33m=[0m[1;34m'purple'[0m[1;33m,[0m[1;33m
    [0m    [0mhighlight_type[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0msorting[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mannotated[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [0mstrfmt[0m[1;33m=[0m[1;32mNone[0m[1;33m,[0m[1;33m
    [0m    [1;33m**[0m[0mkwargs[0m[1;33m,[0m[1;33m
    [0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m     
    Visualize the data and hold all choices as attributes.
    
    Fully customizable through its iniatilization and its attributes
    [1;31mInit docstring:[0m
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
    kwargs 
        Passed to plt.subplots(), e.g. figsize
    
    Raises
    ------
    TypeError
        If data is not of type pd.Series or pd.DataFrame
    [1;31mFile:[0m           c:\users\gebruiker\documents\willekeurige berekeningen\micplot\visualization.py
    [1;31mType:[0m           type
    [1;31mSubclasses:[0m     
    


# `micplot` contains some more useful plottypes
Above, we already saw the bar chart that is often very useful to make a point. Below, we show other plottypes and when `micplot` uses them.

## Waterfall charts for compositions
Waterfall charts are a good choice to show how the total group composition is. Note how `micplot` automatically infers this from the fact that the data contains percentages that add up to 100%.


```python
data = df['Embarked'].value_counts(normalize=True)
micplot.visualize(data)
plt.show()
```


    
![PNG](readme/output_11_0.PNG)
    


## Vertical bars for short timeseries data
Bar charts are the plot of choice for time series data with not too many points. `micplot` infers this from the Index of the data. 
Note how in the plot below the legend is placed outside the plot to prevent the legend from overlapping with the data. The highlight specifies that data point to highlight if there is only one Series, but the column to highlight if multiple Series are compared.


```python
size = 6
columnnames = ['Cars', 'Bikes', 'Buses', 'Planes']
test_data = pd.DataFrame(10*np.random.rand(size, 4), index=pd.date_range('20190101', periods=size), columns=columnnames)
```


```python
micplot.visualize(test_data['Cars'], highlight=-1)
plt.show()
micplot.visualize(test_data, highlight=0)
plt.show()
```


    
![PNG](readme/output_14_0.PNG)
    



    
![PNG](readme/output_14_1.PNG)
    


## Line chart for longer timeseries data
The bar chart would become unreadable if the time series data were longer, so `micplot` changes the plottype to a line plot.


```python
size = 12
test_data = pd.DataFrame(10*np.random.rand(size, 4), index=pd.date_range('20190101', periods=size), columns=columnnames)

micplot.visualize(test_data['Cars'], highlight=-1)
plt.show()
micplot.visualize(test_data, highlight=0)
plt.show()
```


    
![PNG](readme/output_16_0.PNG)
    



    
![PNG](readme/output_16_1.PNG)
    


## Scatter plots to investigate the relationship between two series


```python
micplot.visualize(df[['Age', 'Fare']])
plt.show() 
```


    
![PNG](readme/output_18_0.PNG)
    


If there are only a few datapoints in the plot, the points are also labeled with their index.


```python
micplot.visualize(df[['Age', 'Fare']].sample(15))
plt.show() 
```


    
![PNG](readme/output_20_0.PNG)
    


If there is a third column, this is turned into a bubble chart, where the third column determines the marker size. Here we see that `micplot` gives the legend a title when appropriate.


```python
micplot.visualize(df[['Age', 'Fare', 'Parch']])
plt.show() 
```


    
![PNG](readme/output_22_0.PNG)
    


## Composition comparison can show how subpopulations differ
If we quickly want to infer whether subgroups have the same distribution, we can use a stacked bar chart. `micplot` automatically chooses this if the data to visualize is a DataFrame where each column is a percentage Series.



```python
data = (df.groupby('Pclass')['Survived'].value_counts(normalize=True)
        .unstack(level='Pclass')
       )
display(data)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.37037</td>
      <td>0.527174</td>
      <td>0.757637</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.62963</td>
      <td>0.472826</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>



```python
vis = micplot.visualize(data)
```


    
![PNG](readme/output_25_0.PNG)
    


## Pie chart works as expected


```python
micplot.visualize(df['Embarked'].value_counts(), plottype='pie')
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-1097c721adaa> in <module>
    ----> 1 micplot.visualize(df['Embarked'].value_counts(), plottype='pie')
    

    ~\Documents\Willekeurige berekeningen\micplot\visualization.py in visualize(data, **kwargs)
        693     """
        694     vis = Visualization(data, **kwargs)
    --> 695     vis.plot()
        696     return vis
        697 
    

    ~\Documents\Willekeurige berekeningen\micplot\visualization.py in plot(self)
        638         color = self._define_colors()
        639         linestyles = self._define_linestyles()
    --> 640         self.ax = plotter(self._data_to_plot, color=color, style=linestyles, ax=self.ax)
        641 
        642         if self.annotated:
    

    ~\Documents\Willekeurige berekeningen\micplot\plotfunctions.py in plot_pie(data, **kwargs)
        114 def plot_pie(data, **kwargs):
        115     """ Plot pie chart - small joke."""
    --> 116     raise TypeError('A pie chart? Are you kidding me?')
        117 
        118 def plot_composition_comparison(data, **kwargs):
    

    TypeError: A pie chart? Are you kidding me?



    
![PNG](readme/output_27_1.PNG)
    

