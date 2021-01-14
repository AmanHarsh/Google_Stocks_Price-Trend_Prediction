# -*- coding: utf-8 -*-
###DATA_PREPROCESSING
#Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Loading_Dataa
dataset_train = pd.read_csv('G:\ML\Google Stocks\Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature_Scaling
scale = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scale.fit_transform(training_set)

#Data_Structure with 80TimeSteps
x_train = []
y_train = []
for i in range (80,training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-80:i,0])
    y_train.append(training_set_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

#Adding_Dimension
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))





###CREATING_MODEL
#Import
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initiallization
regressor = Sequential()

#Adding 5-LSTM and Dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Output_Layer
regressor.add(Dense(units=1))

#Compile
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting_Model
regressor.fit(x_train, y_train, epochs=100, batch_size=30, verbose=2)






###TESTING
#Loading_Dataset
dataset_test = pd.read_csv('G:\ML\Google Stocks\Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#Concating Test-Data with Train-Data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

#Generating_Test-Inputs
inputs = dataset_total[len(dataset_total)-len(dataset_test)-80:].values
inputs = inputs.reshape(-1,1)
inputs = scale.transform(inputs)

#Data-Structure for 80Timesteps
x_test = []
for i in range (80, 100):
    x_test.append(inputs[i-80:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

#Predicting
predicted_stock_price = regressor.predict(x_test)    

#Original_Scale
predicted_stock_price = scale.inverse_transform(predicted_stock_price)





###VISUALIZATION
plt.plot(real_stock_price, color='red', label='Real_Stock_Trend')
plt.plot(predicted_stock_price, color='blue', label='Predicted_Stock_Trend')
plt.title('Google_Stock_Trend_Prediction')
plt.xlabel('Time')
plt.ylabel('Google_Stock_Price')
plt.legend()
plt.show()