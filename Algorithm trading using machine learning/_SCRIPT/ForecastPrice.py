# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:45:43 2018

@author: kennedy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as web
from sklearn.model_selection import KFold, GridSearchCV
from subprocess import check_output
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import math
#
start_date = datetime(2000, 1, 1)
end_date = datetime.now()


data = web.DataReader('IBM', "yahoo", start_date, end_date)
forecast_col = 'Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)
X = data.iloc[:, :5]
X = preprocessing.scale(X)
Xf = X[:forecast_out]
X = X[forecast_out:]
data.dropna(inplace=True)
Y = np.array(data['label'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state = 0)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)

fX_set = clf.predict(Xf)
data['Forecast'] = np.nan
last_date = data.index[-1]
last_date = datetime.strptime(str(last_date), '%Y-%m-%d 00:00:00').timestamp()
last_unix = last_date
one_day = 86400
next_unix = last_unix + one_day
next_unix = last_unix + one_day

for i in fX_set:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]
data[['Close', 'Forecast']].plot()
#plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()