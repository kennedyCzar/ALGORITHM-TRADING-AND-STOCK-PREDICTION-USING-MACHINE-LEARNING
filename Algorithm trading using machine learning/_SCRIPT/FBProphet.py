# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:33:59 2018

@author: kennedy

Prerequisites:
    stan: >pip install pyspan
        '''to avoid logger error...
    Cython: >conda install -c anaconda cython
    
    Click on the python symbol in your spyder 
    to append the fbprophet library to your 
    environment path.
    Add the paths: D:\PYTHONPATH\prophet-master\python\fbprophet
                    D:\PYTHONPATH\prophet-master\python\stan
Installation:
    Open anaconda propmt
    >Locate the directory containing your fbprophet setup.py
    for me its
            > D:\PYTHONPATH\prophet-master\python
            > Enter the following command
            >python setup.py install
            ........DONE
            
    Now we are ready to tweak fbprophet.
    We can add anything we want to the library 
    as we may desire.

FINAL NOTE:
    Remember the essence of this is to get
    acquainted with how the facebook team predict 
    future price using timeseries.
    
Resources:
    http://dacatay.com/data-science/part-5-time-series-prediction-prophet-python/
    
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from fbprophet import Prophet
import pandas_datareader.data as web


class FBForecaster(object):
    def __init__(self, stock, feature, start, end, periods_ahead):
        self.start = start
        self.end = end
        self.stock = stock
        self.feature = feature
        self.data = web.DataReader(self.stock, "yahoo", self.start, self.end)
        self.periods_ahead = periods_ahead
        
    def forecast(self):
        self.dS = pd.DataFrame(self.data.index)
        self.y = pd.DataFrame(np.array(self.data.loc[:, [self.feature]]), columns = ['y'])
        self.df = pd.concat([self.dS, self.y], axis = 1)
        self.df.columns = ['ds', 'y']
        self.model = Prophet()
        self.model.fit(self.df)
        self.future = self.model.make_future_dataframe(periods = self.periods_ahead)
        self.forecast = self.model.predict(self.future)
        self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        self.model.plot(self.forecast)



#%% UNIT TEST

dataframe = ['TSLA', 'IBM', 'AAPL', 'MSFT', 'F', 'GM', 'GOLD', 'FB']
    #end_date = datetime(2018, 7, 16)
for ii in dataframe:
    fb = FBForecaster(ii, 'Close', datetime(2000, 1, 1), datetime.now(), 365)
    fb.forecast()
    plt.savefig("../FBPROPHET_IMAGES/FBProphetResult_For_{}_2018.png".format(ii))












