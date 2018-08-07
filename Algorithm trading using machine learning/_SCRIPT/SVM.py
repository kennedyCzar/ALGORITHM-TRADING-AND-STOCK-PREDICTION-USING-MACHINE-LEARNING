# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 21:39:41 2018

@author: kennedy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as web
from sklearn.model_selection import KFold, GridSearchCV
#
start_date = datetime(2016, 1, 1)
end_date = datetime(2018, 7, 16)


data = web.DataReader('IBM', "yahoo", start_date, end_date)


#define the feature vector we would be using for 
#to plot our regression
df = data[['Open']]

df['Volatility'] = df['Open'] - df['Open'].shift(1).fillna(0)

#SVM model
from sklearn.svm import SVR

#this we would be using to draw our regression line
Xf1 = np.arange(1, len(df)+ 1)
#Xf2 = (Xf1**2).astype(np.float64)
#Xf3 = (Xf1**3).astype(np.float64)
#Xf4 = (Xf1**4).astype(np.float64)

#put our numpy array in a list
Xf = [Xf1]#, Xf2]#, Xf3#, Xf4]
#transpose and reshape our data into (Nx4)Dimensions
Xf = np.reshape(Xf, (1, len(df))).T
Yf = df['Open']

#k_fold = KFold(n_splits = 3, random_state=None, shuffle=False)
#for train, test in k_fold.split(Xf):
#    X_train, X_test = Xf[train], Xf[test]
#    Y_train, Y_test = Yf[train], Yf[test]
    #create a regression class
svect_ = SVR(kernel = 'linear', max_iter = 3)
svect_.fit(Xf, Yf)
    #fit grid_
#print('best_estimator_{}'.format(gridSearch_.best_estimator_))
#print('GridScore_{}'.format(gridSearch_.grid_scores_))
#print('bestScore{}'.format(gridSearch_.best_score_))
#print('Test Score:{}'.format(regress.score(X_test, Y_test)))

#get the coefficients and intercept
coeffs = svect_.coef_
intercept = svect_.intercept_

#create a Regression and residual column
#in out dataframe
df['Regression'] = intercept + coeffs[0][0] * Xf1# + coeffs[0][1] * Xf2# + coeffs[2] * Xf3 + coeffs[3] * Xf4
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
#dt3 = dt **3
#dt4 = dt **4

dt_predict = intercept + coeffs[0][0] * dt# + coeffs[0][1] * dt2# + coeffs[2] * dt3 + coeffs[3] * dt4
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
plt.plot(dt_predict, label="Predicted", lw = 1.)
plt.plot(dt_predict - std_regress, label='Upper regress band', lw = 1.)
plt.plot(dt_predict + std_regress, label='Upper regress band', lw = 1.)
plt.legend()
plt.grid(True)
plt.title("Regression forecast")
plt.show()