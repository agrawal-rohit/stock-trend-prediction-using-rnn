# Google Stock Price Predictor

This repository contains a Recurrent Neural Network (RNN) model for predicting Google's stock price. The model is trained on the Google Open Stock Price Dataset and uses Long Short-Term Memory (LSTM) units along with Dropout layers for regularization.

## Methodology

1. **Data Import:** The training dataset is imported from 'data/google-stock-price-train.csv'. The data is then scaled using MinMaxScaler from sklearn.preprocessing.

2. **Data Structure Creation:** A data structure with 60 timesteps and 1 output is created. This means that to predict the next value, the model 'memorizes' the last 60 values.

3. **RNN Building:** The RNN model is built using Keras. It consists of four LSTM layers with 50 units each and a Dropout of 0.2 after each LSTM layer. The final layer is a Dense layer with 1 unit. The model is compiled with 'adam' optimizer and 'mean_squared_error' loss function.

4. **Model Training:** The model is trained with 100 epochs and a batch size of 32.

5. **Prediction:** The model predicts the stock prices for the test data. The predicted values are then transformed back to the original scale.

6. **Visualization:** The real and predicted stock prices are plotted for comparison.

## Results

The model successfully predicts the trend in Google's stock prices. The predicted values closely follow the real stock prices, except for a few anomalies.

![Results](https://github.com/agrawal-rohit/google-stock-trend-predictor/assets/29514438/3b0cbc6b-7fa0-4285-a1ea-5ba14aa6f15b)
