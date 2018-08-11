# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 23:28:08 2018

@author: kennedy
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from keras import optimizers
from datetime import datetime
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split

start_date = datetime(1987, 1, 1)
end_date = datetime(2018, 7, 16)


data = web.DataReader('F', "yahoo", start_date, end_date)


scl = MinMaxScaler()
#scl = StandardScaler()

data = scl.fit_transform(data)

X = data[:, [0, 1, 2, 5]]
Y = data[:, [3]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


X = X.reshape((X.shape[0],X.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

model = Sequential()
model.add(LSTM(200,input_shape=(4,1)))

model.add(Dense(1))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.6, nesterov=True)

model.compile(optimizer='adam',loss='mse')
model.summary()

history = model.fit(X,Y,epochs=40,validation_data=(X_test,Y_test),shuffle=False,batch_size=7)
#history = model.fit(X,Y,epochs = 50,shuffle = False)

plt.plot(history.history['loss'])
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")

predict = model.predict(X_test)
plt.plot(predict,label= "Predictions")
plt.plot(Y_test,label="True Values")
plt.legend()
plt.title("Predictions")
plt.xlabel("Days after training date")
plt.ylabel("Normalized Stock Closing Price")

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
outArray = np.concatenate((X_test,predict),axis=1)
val = scl.inverse_transform(outArray)

predVal = val[:,4]
test = test.astype(float)
trueVal = test[:,4]

plt.plot(predVal,label="Predicted Value")
plt.plot(trueVal, label="True Value")

print(predVal)

plt.legend()
plt.title("True Value vs Predicted Value")
plt.xlabel("Days after training data")
plt.ylabel("Google Stock Value")

plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')

print(min(history.history['val_loss']))
print(min(history.history['loss']))
