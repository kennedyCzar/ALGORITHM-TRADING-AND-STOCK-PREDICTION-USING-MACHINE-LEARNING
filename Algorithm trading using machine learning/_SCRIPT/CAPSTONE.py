# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:29:20 2018

@author: kennedy
"""


import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as snb
from datetime import datetime


start_date = datetime(2000, 1, 1)
end_date = datetime(2018, 7, 16)
df = web.DataReader("IBM", "yahoo", start_date, end_date)

def plot(file):
    import matplotlib.pyplot as plt
    file = file.plot()
    plt.grid()
    return file
    
plot(df)


def plot_main(file):
    import matplotlib.pyplot as plt
    #file['MA50'] = pd.rolling_mean(file['Close'], 50)
    #file['MA200'] = pd.rolling_mean(file['Close'], 200)
    #file['Date'] = pd.to_datetime(file['Date'])
    #file.set_index('Date', inplace = True)
    #file[['Low', 'Close']].plot(subplots = True, figsize = (18, 16))
    file[['Close', 'Open', 'High', 'Low']].plot()
    plt.grid(True)
    plt.title('IBM yearly chart')
    plt.xlabel('year')
    plt.ylabel('closing price')
    plt.show()

plot_main(df)


#%% BOLLIGER BANG AND MOVING AVERAGE

#df['Close'].expanding().mean().plot() #expaned moving average
#
#df['Close'].plot(figsize = (16, 12))
#df.rolling(100).mean()['Close'].plot(figsize = (16, 12))


def bollinger_band(dataframe, average_price):
    import matplotlib.pyplot as plt
    
    dataframe['Close {} day MA'.format(average_price)] = dataframe['Close'].rolling(average_price).mean()
    dataframe['Upper band'] = dataframe['Close {} day MA'.format(average_price)] + 2*(dataframe['Close'].rolling(average_price).std())
    dataframe['Lower band'] = dataframe['Close {} day MA'.format(average_price)] - 2*(dataframe['Close'].rolling(average_price).std())
    dataframe[['Close', 'Upper band', 'Lower band', 'Close {} day MA'.format(average_price)]].plot(lw = 1.)
    plt.grid(True)
    plt.title('Bollinger band of {}'.format(average_price))
    plt.xlabel('year in view')
    plt.ylabel('Prices')
    

bollinger_band(df, 250)

    
    
    
#%%CAPSTONE PROJECT

#getting the data
import pandas_datareader
import datetime
import matplotlib.pyplot as plt

start, end = datetime.datetime(2010, 1, 1), datetime.datetime(2018, 7,1)

tesla = web.DataReader('TSLA', 'yahoo', start, end)
ford = web.DataReader('F', 'yahoo', start, end)
gm = web.DataReader('GM', 'yahoo', start, end)
dax = web.DataReader('DAX', 'yahoo', start, end)

#data visualization
#plot opening prices
def open_prices():
    tesla['Open'].plot(label = 'Tesla', title = 'Opening Prices')
    gm['Open'].plot(label = 'GM')
    ford['Open'].plot(label = 'Ford')
    plt.legend()



#plotting volume
def volume():
    tesla['Volume'].plot(label = 'Tesla', title = 'Opening Prices')
    gm['Volume'].plot(label = 'GM')
    ford['Volume'].plot(label = 'Ford')
    plt.legend()

#maximum numbers of volume traded by ford and day
ford['Volume'].max()
#Out[142]: 480879500.0
ford['Volume'].idxmax()
#Out[141]: Timestamp('2011-01-28 00:00:00')
print(ford.describe().transpose())

#ford['Open'].plot(label = 'Ford')

#plot of total market capitalization
#how much is been traded each day.
def total_market_cap():
    tesla['Total Traded'] = tesla['Open']*tesla['Volume']
    ford['Total Traded'] = ford['Open']*ford['Volume']
    gm['Total Traded'] = gm['Open']*gm['Volume']
    
    tesla['Total Traded'].plot(label = 'Tesla', title = 'Total traded Prices')
    gm['Total Traded'].plot(label = 'GM')
    ford['Total Traded'].plot(label = 'Ford')
    plt.legend()
    
total_market_cap()


def plot_MA(dataframe, first_ma, second_ma, price_option):
    dataframe['MA{}'.format(first_ma)] = dataframe[price_option].rolling(window = first_ma).mean()
    dataframe['MA{}'.format(second_ma)] = dataframe[price_option].rolling(window = second_ma).mean()
    dataframe[[price_option, 'MA{}'.format(first_ma), 'MA{}'.format(second_ma)]].plot( title = 'MA plot of {} over {}'.format(first_ma, second_ma))


plot_MA(tesla, 50, 200, 'Close')

#%%SCATTER MATRIX
#correlation between stocks
from pandas.plotting import scatter_matrix
car_company = pd.concat([tesla['Open'], gm['Open'], ford['Open']], axis = 1)
car_company.columns = ['Tesla Open', 'GM Open', 'Ford Open']
scatter_matrix(car_company)

#%% PREDICTION
def prediction(dataframe, ma1, ma2):
    '''
    Now we do a multivariant regression
    closing_price = beta0 + beta1*X1 + beta2*X2
    beta0--> interceptiong closing price
    beta1-->coefficient of the 3day moving average
    beta2--> coefficient of the 9day moving average
    X1 and X2 are the respective moving averages..ie. 
    the independent variable required to predict the closing price
    '''
    #ma1 = 3
    #ma2 = 9
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.arima_model import ARIMA
    dataframe['MA{}'.format(ma1)] = dataframe['Close'].shift(1).rolling(ma1).mean()
    dataframe['MA{}'.format(ma2)] = df['Close'].shift(1).rolling(ma2).mean()
    X = dataframe[['MA{}'.format(ma1), 'MA{}'.format(ma2)]]
    X = X.dropna()
    cdata = dataframe[['MA{}'.format(ma1), 'MA{}'.format(ma2), 'Close']]
    Y = cdata.dropna()
    Y = Y['Close']
    #X[['MA3', 'MA9']].plot()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
    model = sm.tsa.statespace.SARIMAX(df[['MA{}'.format(ma1), 'MA{}'.format(ma2)]])
    result = model.fit()
    model_fit = LinearRegression().fit(X_train, Y_train)
    print('IBM closing price =', model_fit.intercept_, '+',  round(model_fit.coef_[0], 2),
          '* 3day MA', '+', round(model_fit.coef_[1], 2),
          '* 9day MA')
    
    predicted_price = model_fit.predict(X_test)
    #convert the predicted price into a pandas dataframev
    predicted_price = pd.DataFrame(predicted_price, index = Y_test.index, columns = ['price'])
    predicted_price.plot(title = 'Predicted IBM closing price')
    Y_test.plot()
    plt.grid(True)
    plt.legend(['predicted_price', 'Actual Price'])
    #check the r-squared error
    r2_score = model_fit.score(X_test, Y_test)
    print('R-sqaured error is: {}%'.format(round(r2_score*100, 2)))
    #forecast = int(30)
    #create more time series for future prediction
    #X_test.plot()
    from pandas.tseries.offsets import DateOffset
    X_train['forecast'] = predicted_price
    future_dates = [X_train.index[-1] + DateOffset(months = x) for x in range(1, 30)]
    future_df = pd.DataFrame(index = future_dates, columns = X_train.columns)
    final_df = pd.concat([X_train, future_df])
    
    #forcast or prediction
    final_df['forecast'] = model.predict(start = , end = , dynamic= True)
    final_pred = model.predict(final_df)
    final_df[['Close', 'forecast']].plot(title = 'Final forecast')
    
prediction(tesla, 50, 120)
    
    
    
#%% https://enlight.nyc/stock-market-prediction/

#df_sample = df_sample[['Adj Close']]
#df_sample = df[['Adj Close']]
#df_sample['Prediction'] = df_sample[['Adj Close']]
#df_sample.isnan().any().any()
#df_sample['Prediction'] = df_sample[['Adj Close']].shift(-forecast_out)
#X_sample = np.array(df_sample.drop(['Prediction'], 1))
#X_sample = preprocessing.scale(X_sample)
#from sklearn import preprocessing
#X_sample = preprocessing.scale(X_sample)
#X_forecast = X_sample[-forecast_out:]
#X = X_sample[:-forecast_out]
#X_sample = X_sample[:-forecast_out]
#Y_sample = np.array(X_sample['Prediction'])
#Y_sample = np.array(df_sample['Prediction'])
#Y_sample = Y_sample[:-forecast_out]
#forecast_prediction = model_fit.predict(X_forecast)
#clf = LinearRegression()
#clf.fit(X_sample, Y_sample)
#forecast_prediction = clf.predict(X_forecast)
#print(forecast_prediction)
#forecast_prediction.plot()

    
    
#%% 

#final_df = pd.concat([new_df, fnew_df])
#pred = model_fit.predict(final_df)
#final_df[:, [0, 1]]
#final_df.loc[:, [0, 1]]
#final_df.iloc[:, [0, 1]]
#X_train
#Y_train
#X_train
#final_df.iloc[:, [0, 1]]
#final_df.iloc[:, [0, 1]].shape
#X_anal = final_df.drop(final_df.Close, axis = 1)
#X_anal = final_df.drop(label = final_df.Close, axis = 1)
#X_anal = final_df.drop(columns = ['Close'])
#Y+anal = final_df.drop(colummns = ['MA3', 'MA9'])
#Y_anal = final_df.drop(colummns = ['MA3', 'MA9'])
#Y_anal = final_df.drop(columns = ['MA3', 'MA9'])
#Y_anal
#clf = LinearRegression().fit(X_anal, Y_anal)
#X_anal[:, ].shape
#X_anal.iloc[:, ].shape
#X_anal.iloc[:2644, ].shape
#X_anal.iloc[:2644, ].tail()
#X_anal.iloc[:2645, ].shape
#X_anal.iloc[:2645, ].tail()
#X_anal.iloc[:2644, ].tail()
#X_anal.iloc[:2644, ].shape()
#X_anal.iloc[:2644, ].shape
#X_anal.iloc[:2644, ].tail()
#Y_anal.iloc[:2644, ].tail()
#clf = LinearRegression().fit(X_anal.iloc[:2644, ], Y_anal.iloc[:2644, ])
#clf
#pred_new = clf.predict(fnew_df)


#%% SIGNAL GENERATOR



def signal_gnerator(dataframe, short_price, long_price):
    '''
    Arguments:
        
        dataframe: Dataset to be used for signal generation
        short_price: shorting moving window value for moving average(int)
        long_price: longing moving window value for moving average(int)
    '''
    import matplotlib.pyplot as plt
    import matplotlib.fin

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=dataframe.index)
    signals['Signal'] = 0.0

    # Create short simple moving average over the short window
    signals['Short_MA'] = dataframe['Close'].rolling(window = short_price, min_periods=1, center=False).mean()
    # Create long simple moving average over the long window
    signals['Long_MA'] = dataframe['Close'].rolling(window = long_price, min_periods=1, center=False).mean()

    # Create signals
    signals['Signal'][short_price:] = np.where(signals['Short_MA'][short_price:] > signals['Long_MA'][short_price:], 1.0, 0.0)   

    # Generate trading orders
    signals['Positions'] = signals['Signal'].diff()

    # Initialize the plot figure
    fig = plt.figure()
    
    # Add a subplot and label for y-axis
    ax = fig.add_subplot(111,  ylabel='Price in USD$')
    # Plot the closing price
    dataframe['Close'].plot(title = 'Signal generator using {}MA and {}MA'.format(short_price, long_price), ax = ax, lw = 1.)
    
    # Plot the short and long moving averages
    signals[['Short_MA', 'Long_MA']].plot(ax = ax, lw = 1.)
    
    # Plot the buy signals
    ax.plot(signals.loc[signals.Positions == 1.0].index,
             signals.Short_MA[signals.Positions == 1.0],
             '^', markersize=10, color='g')
    
    # Plot the sell signals
    ax.plot(signals.loc[signals.Positions == -1.0].index, 
             signals.Long_MA[signals.Positions == -1.0],
             'v', markersize=10, color='r')
    
    #add a grid
    plt.grid(True)
    # Show the plot
    plt.show()


signal_gnerator(df, 50, 120)


'''END'''
#%%  Regression line

import matplotlib.pyplot as plt

#define data to use
Xf = web.DataReader("IBM", "yahoo", "2000-01-01", "2018-07-16")

#use Close price and volatility
#Xf['Volatility'] = Xf['Close'] - Xf['Close'].shift(1)
Xf = Xf.dropna()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#data preprocessing
#Xf_wtclose = Xf.drop(['Close'], axis = 1)
#scale_Xf = StandardScaler().fit(Xf_wtclose)
#X_new = scale_Xf.transform(Xf_wtclose)

#regression
regress = LinearRegression()
Xf_wtclose = Xf.drop(['Close'], axis = 1)
regress.fit(Xf_wtclose[['Adj Close']], Xf['Close'])
coeffs = regress.coef_
intercept = regress.intercept_

Xf['Regression'] = intercept + coeffs[0] * Xf['High'] + coeffs[1] * Xf['Low'] + coeffs[2] * Xf['Open'] + coeffs[3] * Xf['Volume'] + coeffs[4] * Xf['Adj Close']
Xf['Prediction'] = intercept + regress.predict(Xf_wtclose[['Adj Close']])
std_close = Xf['Close'].std()

Xf['upper_regress'] = Xf['Regression'] + Xf['Regression'].std()
Xf['lower_regress'] = Xf['Regression'] - Xf['Regression'].std()
#plt.figure(figsize = (18, 15))
ax = plt.subplot(111)
Xf[['Close', 'Prediction']].plot()

#----------------
import statsmodels.api as sm
from scipy import mstats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
X, Y = df[['Open']], df[['Close']]
X = sm.add_constant(X)
slope, intercept, r_value, p_value, std_err = stats.linregress(X.iloc[:, 0], Y.iloc[:, 0])

yhat = slope*X.iloc[:, 0] + intercept #this is the regression line
Xf['yhat'] = yhat
Xf[['Close', 'yhat']].plot()
yhat.plot()




model = sm.OLS(Y,X)
result = model.fit()
print(result.summary())
prstd, lower, upper = wls_prediction_std(result)
Xf[['Close', prstd, lower, upper]].plot()
prstd = pd.DataFrame(prstd, index = Xf.index, columns = ['prstd'])
lower = pd.DataFrame(lower, index = Xf.index, columns = ['lower'])
upper = pd.DataFrame(upper, index = Xf.index, columns = ['upper'])
Xf['prstd'] = prstd
Xf['lower'] = lower
Xf['upper'] = upper
Xf[['Close', 'prstd', 'lower', 'upper']].plot()

#%% REGRESSION LINE

#import statsmodels.api as sm
#import matplotlib.pyplot as plt
#
#
#start_date = datetime(2000, 1, 1)
#end_date = datetime(2018, 7, 16)
#X = web.DataReader("IBM", "yahoo", start_date, end_date)
#
#X['day'] = X.index.map(lambda observation_timestamp: observation_timestamp.dayofyear)
#Y = df.Close
#Xf = X.day
#Xf = sm.add_constant(Xf)
#model = sm.OLS(Y, Xf).fit()
#
#X_prime = np.linspace(Xf.day.min(), Xf.day.max(), 100)
#X_prime = sm.add_constant(X_prime)
#yhat = model.predict(X_prime)
#pd_yhat = pd.DataFrame(yhat, Xf[:100].index)
#pd_xprime = pd.DataFrame(X_prime[:, 1], Xf[:100].index)
#
#
#
#plt.plot(Xf.day[:100], Y[:100])
#plt.plot(pd_xprime, pd_yhat)
#sm.stats.linear_rainbow(model)
#sm.graphics.plot_regress_exog(model, "day")
#sm.graphics.plot_fit(model, "day")
#plt.scatter(X.day, Y)
#plt.show()

#%% REGRESSION ON ANY STOCK DATA

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime
import pandas_datareader.data as web

 
def Regression(dataframe, feature, start_date, end_date):
    
    '''
    Arguments:
        dataframe: dataframe we are loading
        dataname: This is the required name of the company deta we want to perform
                    regression
        feature: It indicates the dependent variable we would be prediction
        start_date: As implied signifies the start date of the stock we intend to predict
        end_date:   As implied signifies the end date of the stock we intend to predict
        
    '''
    
    #Load data using datetime for the
    #start and end date
    #start_date = datetime(1976, 1, 1)
    #end_date = datetime(2018, 7, 16)
    try:
        if start_date == None and end_date == None:
            raise('No data to plot regression')
        else:
            data = web.DataReader(dataframe, "yahoo", start_date, end_date)
    except ValueError:
        raise
    finally:
        pass
    
    #define the feature vector we would be using for 
    #to plot our regression
    df = data[[feature]]
    
    #Its a dataframe so we have to convert
    #it into a numerical data
    #We also dont need this since the data is already in float
    #df.info() to check datatype
    #df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    
    df['Volatility'] = df[feature] - df[feature].shift(1)
    df = df.dropna()
    
    #linear regression model
    from sklearn.linear_model import LinearRegression
    
    #this we would be using to draw our regression line
    Xf1 = np.arange(1, len(df)+ 1)
    Xf2 = (Xf1**2).astype(np.float64)
    Xf3 = (Xf1**3).astype(np.float64)
    Xf4 = (Xf1**4).astype(np.float64)
    
    #put our numpy array in a list
    Xf = [Xf1, Xf2, Xf3, Xf4]
    #transpose and reshape our data into (Nx4)Dimensions
    Xf = np.reshape(Xf, (4, len(df))).T
    
    
    #create a regression class
    regress = LinearRegression(n_jobs = -1)
    regress.fit(Xf, df[feature])
    
    #get the coefficients and intercept
    coeffs = regress.coef_
    intercept = regress.intercept_
    
    #create a Regression and residual column
    #in out dataframe
    df['Regression'] = intercept + coeffs[0] * Xf1 + coeffs[1] * Xf2 + coeffs[2] * Xf3 + coeffs[3] * Xf4
    df['Residuals'] = df[feature] - df['Regression'] #Not needed now untill further analysis is required.
    df['Upper regresss bound'] = df['Regression'] + df['Regression'].std()
    df['Lower regresss bound'] = df['Regression'] - df['Regression'].std()
    
    
    df[[feature, 'Regression', 'Upper regresss bound', 'Lower regresss bound',]].plot(title = 'Polynomial Regression line for {} from {} to {}'.format(dataframe, start_date, end_date))
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.show()


#Unit testing
Regression('TSLA', feature = 'Close', start_date = datetime(2016, 1, 1), end_date = datetime(2018, 7, 16))


#Data
#
#tesla = web.DataReader('TSLA', 'yahoo', start, end)
#ford = web.DataReader('F', 'yahoo', start, end)
#gm = web.DataReader('GM', 'yahoo', start, end)
#dax = web.DataReader('DAX', 'yahoo', start, end)
#if __name__ == '__main__':
#    try:
#        if start_date == None and end_date == None:
#            raise('No data to plot regression')
#        else:
#            call_class = regressioon_plot(dataframe, feature, start_date, end_date)
#    except ValueError:
#        raise
#    finally:
#        pass































