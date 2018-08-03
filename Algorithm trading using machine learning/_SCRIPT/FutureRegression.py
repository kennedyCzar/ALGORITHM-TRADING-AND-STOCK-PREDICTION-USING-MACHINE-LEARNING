# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 22:40:18 2018

@author: kennedy
"""

def forecast(dataframe, feature, start_date, end_date, new_end_date):
    
    '''
    Arguments:
        dataframe: list of dataframe we are loading
        dataname: This is the required name of the company deta we want to perform
                    regression
        feature: It indicates the dependent variable we would be prediction
        start_date: As implied signifies the start date of the stock we intend to predict
        end_date:   As implied signifies the end date of the stock we intend to predict
        
    '''
    #Import required libaries
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    import pandas_datareader.data as web
    from datetime import datetime
    
    #-----------------------------
    #start_date = datetime(1976, 1, 1)
    #end_date = datetime(2018, 7, 16)
    #dataframe = ['TSLA', 'IBM', 'AAPL', 'MSFT', 'F', 'GM']
    

    data = web.DataReader(dataframe, "yahoo", start_date, end_date)
    #define the feature vector we would be using for 
    #to plot our regression
    df = data[[feature]]
    

    
    df['Volatility'] = df[feature] - df[feature].shift(1)
    df = df.dropna()
    
    #linear regression model
    from sklearn.linear_model import LinearRegression
    
    #this we would be using to draw our regression line
    Xf1 = np.arange(1, len(df)+ 1)
    Xf2 = (Xf1**2).astype(np.float64)
    Xf3 = (Xf1**3).astype(np.float64)
    #Xf4 = (Xf1**4).astype(np.float64)
    
    #put our numpy array in a list
    Xf = [Xf1, Xf2, Xf3]#, Xf4]
    #transpose and reshape our data into (Nx4)Dimensions
    Xf = np.reshape(Xf, (3, len(df))).T
    
    
    #create a regression class
    regress = LinearRegression(n_jobs = -1)
    regress.fit(Xf, df[feature])
    
    #get the coefficients and intercept
    coeffs = regress.coef_
    intercept = regress.intercept_
    
    #create a Regression and residual column
    #in out dataframe
    df['Regression'] = intercept + coeffs[0] * Xf1 + coeffs[1] * Xf2 + coeffs[2] * Xf3# + coeffs[3] * Xf4
    df['Residuals'] = df[feature] - df['Regression'] #Not needed now untill further analysis is required.
    std_regress = df['Regression'].std()
    std_open = df[[feature]].std()
    
    
    
    #plot future price
    #----------------------------------------------
    #new_end_date = datetime(2020, 7, 16)
    dates = pd.bdate_range(start_date, new_end_date)
    dt = np.arange(1, len(dates) + 1)
    dt2 = dt **2
    #dt3 = dt **3
    #dt4 = dt **4
    
    dt_predict = intercept + coeffs[0] * dt + coeffs[1] * dt2# + coeffs[2] * dt3 + coeffs[3] * dt4
    dt_predict = pd.DataFrame(data = dt_predict, index = dates)
    dt_predict.index.name = 'Date'
    dt_predict.columns = [[feature]]
    actual = data['Open']
    plt.figure(figsize=(18, 16))
    plt.plot(actual, label="Actual")
    plt.plot(dt_predict, label="Predicted")
    plt.plot(dt_predict - std_regress, label='Upper regresss bound')
    plt.plot(dt_predict + std_regress, label='lower regresss bound')
    plt.legend(loc='best')
    plt.title("{} REGRESSION FORECAST FOR {}".format(dataframe, new_end_date))
    #plt.savefig("../_REGRESSION IMAGES/best_2018.png")
    plt.show()
    #----------------------------------------------------

dataframe = ['TSLA', 'IBM', 'AAPL', 'MSFT', 'F', 'GM']
#end_date = datetime(2018, 7, 16)
forecast("TSLA", 'Close', datetime(2012, 1, 1), datetime.now(), datetime(2020, 7, 16))