# **Introduction**

In this session, we will practice the capital asset pricing model using daily stocks of some of the most popular firms in the U.S.
We start off by importing some important libraries and go step by step in creating CAPM model. The Capital Asset Pricing Model (CAPM) is a financial model that seeks to explain the relationship between the risk and return of an asset. It is used to estimate the expected return on an investment based on the risk-free rate, the expected market return, and the asset's beta (a measure of the asset's systematic risk).

According to CAPM, the expected return on an asset is equal to the risk-free rate plus a premium for the asset's systematic risk. The premium is calculated by multiplying the market risk premium (the difference between the expected market return and the risk-free rate) by the asset's beta.

The CAPM model is commonly used in finance to determine the expected return on an investment and to evaluate the performance of investment portfolios. It assumes that investors are rational and risk-averse and that they require compensation for taking on additional risk. However, the model has been subject to criticism, and alternative models have been proposed to address its limitations.


1. [Import data and python packages](#t1.)
2. [Inspect the dataset and Normalize variables](#t2.)
3. [Exploratory Data Analysis using interactive graphs](#t3.)
4. [Create a daily return](#t4.)
5. [Create the CAPM equation and calculate the expected return](#t5.)
    * 5.1. [Graph the CAPM](#t5.1.)


<a id="t1."></a>
# 1. Import Data & Python Packages


```python
import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
```


```python
from jupyterthemes import jtplot # Jupyter theme
jtplot.style(theme = 'monokai', context= 'notebook', ticks= True, grid= False)
```


```python
stocks_df = pd.read_csv(r'../input\stocks_dataset.csv')
```

<a id="t2."></a>
# 2. Inspect the dataset and Normalize variables


```python
stocks_df.head()
```


```python
stocks_df.tail()
```


```python
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x
```


```python
stock_norm = normalize(stocks_df)
```


```python
stock_norm
```

<a id="t3."></a>
# 3. Exploratory Data Analysis using interactive graphs


```python
def interactive_plot(df, title):
    fig = px.line(title = title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
    fig.show()
```


```python
interactive_plot(stocks_df, "Raw Prices")
```


```python
interactive_plot(stock_norm, "Normalized prices")
```

You bought $1000 worth share of Tsla in November 7th, 2013. How much does your stock worth in Novermber 7th 2019?

1000x 2.4 = $2400

<a id="t4."></a>
# 4. Create a daily return


```python
def daily_ret(df):
    df_daily_ret = df.copy()
    for i in df.columns[1:]:
        for j in range(1, len(df)):
            df_daily_ret[i][j] = ((df[i][j] - df[i][j-1]) / df[i][j-1])*100
            df_daily_ret[i][0] = 0
    return df_daily_ret
```


```python
stock_daily_ret = daily_ret(stocks_df)
```


```python
stock_daily_ret
```


```python
stock_daily_ret['NFLX']
```

<a id="t5."></a>
# 5. Create the CAPM equation and calculate the expected return


```python
beta, alpha = np.polyfit(stock_daily_ret['sp500'], stock_daily_ret['NFLX'], 1)
print('Beta for {} stock is = {} and alpha is {}' .format('NFLX', beta, alpha))
```


```python
beta
```

### CAPM MODEL CALCULATION

#### Ri = Rf + Bi (Rm - Rf)


```python
# let's assume that the rf is the U.S. 10-year treasury bill

rf= 2

stock_daily_ret['sp500'].mean()
```


```python
rm = 0.04457361768265508 * 252
```


```python
rm
```


```python
rm = stock_daily_ret['sp500'].mean() * 252
```


```python
rm
```


```python
Exp_retn = rf + (beta * (rm - rf))
```


```python
Exp_retn
```

<a id="t5.1."></a>
## 5.1.   Graph the CAPM


```python
beta = {}
alpha = {}

for i in stock_daily_ret.columns:
    if i != 'Date' and i != 'sp500':
        stock_daily_ret.plot(kind='scatter', x='sp500', y=i, color= 'r')
        b, a = np.polyfit(stock_daily_ret['sp500'], stock_daily_ret[i], 1)
        
        beta[i] = b
        alpha[i] = a
        
        plt.show()
```


```python
beta
```


```python
alpha
```
