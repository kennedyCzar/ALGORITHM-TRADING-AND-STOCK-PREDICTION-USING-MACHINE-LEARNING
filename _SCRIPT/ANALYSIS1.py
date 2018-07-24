# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:55:50 2018

@author: kennedy
"""

import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as snb
snb.pairplot(dataset)
os.chdir('D:\\_TEMP\\_CURRENCY_DATA')
#columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
dataset = pd.read_csv('EURUSD60.csv')

dataset.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']

#write a function to load data and add columns
def loaddata(file):
    #load file directory
    os.chdir('D:\\_TEMP\\_CURRENCY_DATA')
    data = pd.read_csv(file)
    data.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    return data

dataset = loaddata('EURUSD60.csv')

#plotfunction for all the data
def plot(file):
    import matplotlib.pyplot as plt
    return dataset.plot()
    
plot(dataset)

#Now a plot function for just the closing prince
def plot_main(file):
    import matplotlib.pyplot as plt
    #file['MA50'] = pd.rolling_mean(file['Close'], 50)
    #file['MA200'] = pd.rolling_mean(file['Close'], 200)
    file['Date'] = pd.to_datetime(file['Date'])
    file.set_index('Date', inplace = True)
    file[['Open', 'High', 'Low', 'Close']].plot()
    plt.title('EURUSD yearly chart')
    plt.xlabel('year')
    plt.ylabel('closing price')
    plt.show()

plot_main(dataset)



dataset['Date'] = pd.to_datetime(dataset['Date'])
#dataset['Date'].apply(pd.to_datetime)
dataset.set_index('Date', inplace = True)

dataset['MA30'] = dataset['Close'].rolling(30).mean()
dataset['MA50'] = dataset['Close'].rolling(50).mean()
dataset[['MA30', 'MA50','Close']].plot()


#%%
import statsmodels.api as sm
#X_con = X.add_constant(X)
for r in range(len(dataset)):
    model = sm.OLS(dataset.Close, dataset.index[r]).fit()
print(model.summary())


beta, beta0, r_value, p_value, std_err = stats.linregress(dataset.index, dataset.Close)
