# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:31:35 2018

@author: kennedy
"""


import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as snb
from datetime import datetime
import matplotlib.pyplot as plt


#def bollinger_band(data, average_price):
start_date = datetime(2012, 1, 1)
#end_date = datetime(2018, 7, 16)
end_date = datetime.now()

data = web.DataReader('TSLA', "yahoo", start_date, end_date)
average_price = 50

data['Close {} day MA'.format(average_price)] = data['Close'].rolling(average_price).mean()
data['Upper band'] = data['Close {} day MA'.format(average_price)] + 2*(data['Close'].rolling(average_price).std())
data['Lower band'] = data['Close {} day MA'.format(average_price)] - 2*(data['Close'].rolling(average_price).std())
data['Volatility'] = (data['Close'] - data['Close'].shift(1)).fillna(0)
#BOLLINGER SIGNAL GENERATOR


#plot chart
data[['Close', 'Upper band', 'Lower band', 'Close {} day MA'.format(average_price), 'Volatility']].plot(lw = 1.)
plt.grid(True)
plt.title('Bollinger band of {}'.format(average_price))
plt.xlabel('year in view')
plt.ylabel('Prices')

#
#bollinger_band(df, 250)