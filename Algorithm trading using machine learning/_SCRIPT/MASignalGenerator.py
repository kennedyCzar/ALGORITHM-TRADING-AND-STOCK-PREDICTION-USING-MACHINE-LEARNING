# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:49:49 2018

@author: kennedy
"""


import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as snb
from datetime import datetime

#
#start_date = datetime(2000, 1, 1)
##end_date = datetime(2018, 7, 16)
#end_date = datetime.now()
#df = web.DataReader("AAPL", "yahoo", start_date, end_date)


def signal_gnerator(stock_name, short_price, long_price, start_date, end_date):
    '''
    Arguments:
        
        dataframe: Dataset to be used for signal generation
        short_price: shorting moving window value for moving average(int) short_price = 70
        long_price: longing moving window value for moving average(int) long_price = 250
    '''
    import matplotlib.pyplot as plt
    
    #load data
    dataframe = web.DataReader(stock_name, "yahoo", start_date, end_date)
    #get volatility
    dataframe['Volatility'] = (dataframe['Close'] - dataframe['Close'].shift(1)).fillna(0)
    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=dataframe.index)
    signals['Signal'] = 0.0

    # Create short simple moving average over the short window
    signals['Long_MA'] = dataframe['Close'].rolling(window = short_price, min_periods=1, center=False).mean()
    # Create long simple moving average over the long window
    signals['Short_MA'] = dataframe['Close'].rolling(window = long_price, min_periods=1, center=False).mean()

    # Create signals
    signals['Signal'][short_price:] = np.where(signals['Short_MA'][short_price:] > signals['Long_MA'][short_price:], 1.0, 0.0)   

    # Generate trading orders
    signals['Positions'] = signals['Signal'].diff()

    # Initialize the plot figure
    fig = plt.figure()
    
    # Add a subplot and label for y-axis
    ax = fig.add_subplot(111,  ylabel='Price in USD$')
    # Plot the closing price
    dataframe['Close'].plot(title = '{} Signal generator using {}MA and {}MA'.format(stock_name, short_price, long_price), ax = ax, lw = 1.)
    
    # Plot the short and long moving averages
    signals[['Short_MA', 'Long_MA']].plot(ax = ax, lw = 1.)
    
    dataframe['Volatility'].plot()
    # Plot the buy signals
    ax.plot(signals.loc[signals.Positions == -1.0].index,
             signals.Long_MA[signals.Positions == -1.0],
             '^', markersize=6, color='g')
    
    # Plot the sell signals
    ax.plot(signals.loc[signals.Positions == 1.0].index, 
             signals.Short_MA[signals.Positions == 1.0],
             'v', markersize=6, color='r')
    
    #add a grid
    plt.grid(True)
    # Show the plot
    plt.legend()
    plt.show()


signal_gnerator('MSFT', 70, 250, datetime(2000, 1, 1), datetime.now())
