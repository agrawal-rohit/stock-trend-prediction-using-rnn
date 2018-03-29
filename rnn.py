# Recurrent Neural Networks

# Part 1 : Data Processing( DIfferent from ANN one)

#libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # upper bound of range is excluded, put in range as we need the data as a numpy array which can be put into the network, else it will be a single vector

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # Normal Scaler, Better here since its an RNN and since we have sigmoid function as activation function in the output layer
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []

for i in range(60,1258): # 60th row to last row
    x_train.append(training_set_scaled[i-60:i,0]) # to 'memorize' last 60 row values to predict the next value
    y_train.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

# Reshaping (used when you want to include other predictors too)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # The RNN takes the input as as a 3D tensor with following params => (batch_size or total number of obvs, timesteps, input_dims or no. of predictors)

# Part 2 : Building the RNN
#--------------------------

#libs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initializing the RNN
regressor = Sequential()

#Adding first LSTM layer and Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1],1))) # keep units value high to implement high dimensionality
regressor.add(Dropout(0.2))

#Adding second LSTM layer and Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding third LSTM layer and Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding fourth LSTM layer and Dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling
regressor.compile(optimizer = 'adam', loss='mean_squared_error')

#Fitting
regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)

# Part 3 : Making the predictions and visualizing the results
#------------------------------------------------------------
 
# Getting real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60,80): # 60th row to last row
    x_test.append(inputs[i-60:i,0])
x_test= np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # To get the predicted values in original scale

# Visualizing the results
plt.plot(real_stock_price, color='red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()