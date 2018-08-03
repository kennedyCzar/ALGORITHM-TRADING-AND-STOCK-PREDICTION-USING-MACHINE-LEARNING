# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:28:44 2018

@author: kennedy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as web

#
start_date = datetime(2012, 1, 1)
end_date = datetime(2018, 7, 16)


data = web.DataReader('TSLA', "yahoo", start_date, end_date)


#define the feature vector we would be using for 
#to plot our regression
df = data[['Open']]

#Its a dataframe so we have to convert
#it into a numerical data
#We also dont need this since the data is already in float
#df.info() to check datatype
#df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

df['Volatility'] = df['Open'] - df['Open'].shift(1)
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
regress.fit(Xf, df['Open'])

#get the coefficients and intercept
coeffs = regress.coef_
intercept = regress.intercept_

#create a Regression and residual column
#in out dataframe
df['Regression'] = intercept + coeffs[0] * Xf1 + coeffs[1] * Xf2 + coeffs[2] * Xf3# + coeffs[3] * Xf4
df['Residuals'] = df['Open'] - df['Regression'] #Not needed now untill further analysis is required.
std_regress = df['Regression'].std()
std_open = df['Open'].std()
#df['Upper regresss bound'] = df['Regression'] + (df['Regression'].std())
#df['Lower regresss bound'] = df['Regression'] - (df['Regression'].std())


#plot future price
#----------------------------------------------
end_date1 = datetime(2020, 7, 16)
dates = pd.bdate_range(start_date, end_date1)
dt = np.arange(1, len(dates) + 1)
dt2 = dt **2
dt3 = dt **3
#dt4 = dt **4

dt_predict = intercept + coeffs[0] * dt + coeffs[1] * dt2 + coeffs[2] * dt3# + coeffs[3] * dt4
dt_predict = pd.DataFrame(data=dt_predict, index=dates)
actual = data['Open']
#df['predicted'] = dt_predict
#df['Upper regresss bound'] = df['predicted'] + (df['Regression'].std())
#df['Lower regresss bound'] = df['predicted'] - (df['Regression'].std())
#df['Actual'] = df[['Open']]
#df[['Actual', 'predicted', 'Upper regresss bound', 'Lower regresss bound']].plot(lw = 1., title = 'Say nothing')
#plt.grid(True)
#plt.legend()
#plt.show()

plt.figure(figsize=(18, 16))
plt.plot(actual, label="Actual")
plt.plot(dt_predict, label="Predicted")
plt.plot(dt_predict - std_regress, label='Upper regress band')
plt.plot(dt_predict + std_regress, label='Upper regress band')
plt.legend(loc='best')
plt.title("Blah Blah Blah")
plt.savefig("Predictions_2018.png")
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as web

#
start_date = datetime(2012, 1, 1)
end_date = datetime(2018, 7, 16)


data = web.DataReader('TSLA', "yahoo", start_date, end_date)



















