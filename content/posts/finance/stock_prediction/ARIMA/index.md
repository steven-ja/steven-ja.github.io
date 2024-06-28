---
title: "MSFT Stock Prediction using LSTM or GRU"
date: 2024-06-16T00:00:00+01:00
description: "Short Stock price analysis on MSFT, then a prediction is tested using GRU"
menu:
  sidebar:
    name: GRU
    identifier: GRU
    parent: stock_prediction 
    weight: 9
hero: images/stock-market-prediction-using-data-mining-techniques.jpg
tags: ["Finance", "Deep Learning", "Forecasting"]
categories: ["Finance"]
---
## Introduction

In this article, we will explore time series data extracted from the **stock market**, focusing on prominent technology companies such as Apple, Amazon, Google, and Microsoft. Our objective is to equip data analysts and scientists with the essential skills to effectively manipulate and interpret stock market data.

To achieve this, we will utilize the *yfinance* library to fetch stock information and leverage visualization tools such as Seaborn and Matplotlib to illustrate various facets of the data. Specifically, we will explore methods to analyze stock risk based on historical performance, and implement predictive modeling using **GRU/ LSTM** models.

Throughout this tutorial, we aim to address the following key questions:

1. How has the stock price evolved over time?
2. What is the average **daily return** of the stock?
3. How does the **moving average** of the stocks vary?
4. What is the **correlation** between different stocks?
5. How can we forecast future stock behavior, exemplified by predicting the closing price of Apple Inc. using LSTM or GRU?"

---

## Getting Data

The initial step involves **acquiring and loading** the data into memory. Our source of stock data is the **Yahoo Finance** website, renowned for its wealth of financial market data and investment tools. To access this data, we'll employ the **yfinance** library, known for its efficient and Pythonic approach to downloading market data from Yahoo. For further insights into yfinance, refer to the article titled [Reliably download historical market data from with Python](https://aroussi.com/post/python-yahoo-finance).

### Install Dependencies

```bash
pip install -qU yfinance seaborn
```

### Configuration Code

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
%matplotlib inline #comment if you are not using a jupyter notebook

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime

# Get Microsoft data
data = yf.download("MSFT", start, end)
```

## Statistical Analysis on the price

### Summary

```python
# Summary Stats
data.describe()
```

### Closing Price

The closing price is the last price at which the stock is traded during the regular trading day. A stock’s closing price is the standard benchmark used by investors to track its performance over time.

```python
plt.figure(figsize=(14, 5))
plt.plot(data['Adj Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price [$]')
plt.title('Stock Price History')
plt.legend()
plt.show()
```

### Volume of Sales

Volume is the amount of an asset or security that _changes hands over some period of time_, often over the course of a day. For instance, the stock trading volume would refer to the number of shares of security traded between its daily open and close. Trading volume, and changes to volume over the course of time, are important inputs for technical traders.

```python
plt.figure(figsize=(14, 5))
plt.plot(data['Volume'], label='Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Price History')
plt.show()
```

### Moving Average

The moving average (MA) is a simple **technical analysis** tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks, or any time period the trader chooses.

```python
ma_day = [10, 20, 50]

# compute moving average (can be also done in a vectorized way)
for ma in ma_day:
  column_name = f"{ma} days MA"
  data[column_name] = data['Adj Close'].rolling(ma).mean()

plt.figure(figsize=(14, 5))
data[['Adj Close', '10 days MA', '20 days MA', '50 days MA']].plot()
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Price History')
plt.show()
```

## Statistical Analysis on the returns

Now that we've done some baseline analysis, let's go ahead and dive a little deeper. We're now going to analyze the risk of the stock. In order to do so we'll need to take a closer look at the daily changes of the stock, and not just its absolute value. Let's go ahead and use pandas to retrieve teh daily returns for the **Microsoft** stock.

```python
# Compute daily return in percentage
data['Daily Return'] = data['Adj Close'].pct_change()

# simple plot
plt.figure(figsize=(14, 5))
data['Daily Return'].hist(bins=50)
plt.title('MSFT Daily Return Distribution')
plt.xlabel('Daily Return')
plt.show()

# histogram
plt.figure(figsize=(8, 5))
data['Daily Return'].plot()
plt.title('MSFT Daily Return')
plt.show()

```

## Data Preparation

```python
# Create a new dataframe with only the 'Close column 
X = data.filter(['Adj Close'])
# Convert the dataframe to a numpy array
X = X.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(X)*.95))

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(X)

scaled_data
```

Split training data into small chunks to ingest into LSTM and GRU

```python
# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
seq_length = 60
for i in range(seq_length, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= seq_length+1:
        print(x_train)
        print(y_train, end="\n\n")
  
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

## GRU

Gated-Recurrent Unit (GRU) is adopted in this part

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

lstm_model = Sequential()
lstm_model.add(GRU(units=128, return_sequences=True, input_shape=(seq_length, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(GRU(units=64, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x_train, y_train, epochs=10, batch_size=8)
```

## LSTM

Long Short-Term Memory (LSTM) is adopted in this part

```python
from tensorflow.keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(units=128, return_sequences=True, input_shape=(seq_length, 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=64, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x_train, y_train, epochs=10, batch_size=8)
```

## Testing Metrics

* root mean squared error (RMSE)

```python

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
  
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions_gru = gru_model.predict(x_test)
predictions_gru = scaler.inverse_transform(predictions_gru)
predictions_lstm = lstm_model.predict(x_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# Get the root mean squared error (RMSE)
rmse_lstm = np.sqrt(np.mean(((predictions_lstm - y_test) ** 2)))
rmse_gru = np.sqrt(np.mean(((predictions_gru - y_test) ** 2)))
print(f"LSTM RMSE: {rmse_lstm:.4f}, GRU RMSE: {rmse_gru:.4f}")
```

> "LSTM RMSE: **4.2341**, GRU RMSE: 3.3575"

### Test Plot

{{< img src="/posts/finance/stock_prediction/GRU/images/test_results.png" align="center" title="Results">}}
GRU-based model shows a bit better results both graphically and on MSE. However, this does not tell us anything about the actual profitability of these models.

## Possible trading performance

The strategy implementation is:

* BUY: if prediction > actual_price
* SELL: if prediction < actual_price

To close a position the next candle _close_ is waited. However, LSTM and GRU has some offset that does not allow a proper utilization of this strategy.

Hence, the **returns** of the predictions are adopted.

```python
# Assume a trading capital of $10,000
trading_capital = 10000
pred_gru_df = pd.DataFrame(predictions_gru, columns=['Price'])
pred_test_df = pd.DataFrame(y_test, columns=['Price'])
pred_gru_df['returns'] = pred_gru_df.pct_change(-1)
pred_test_df['returns'] = pred_test_df.pct_change(-1)

# Compute Wins
wins = ((pred_gru_df.dropna().returns<0) & (pred_test_df.dropna().returns<0)) | ((pred_gru_df.dropna().returns>0) & (pred_test_df.dropna().returns>0))
print(wins.value_counts())

returns_df = pd.concat([pred_gru_df.returns, pred_test_df.returns], axis=1).dropna()
total_pos_return = pred_test_df.dropna().returns[wins].abs().sum()
total_neg_return = pred_test_df.dropna().returns[np.logical_not(wins)].abs().sum()

# compute final capital and compare with BUY&HOLD strategy
final_capital = trading_capital*(1+total_pos_return-total_neg_return)
benchmark_return = (valid.Close.iloc[-1] - valid.Close.iloc[0])/valid.Close.iloc[0]
bench_capital = trading_capital*(1+benchmark_return)
print(final_capital, bench_capital)
```

> returns
>
>
> True     81
> False    72
>
> Name: count, dtype: int64

> 10535.325 9617.617

## Conclusion

As showed in the previous section, these two simple Deep Learning models exhibits interesting positive results both regarding regression and trading metrics.
The latter is particularly important, indeed a return of **5%** is obtained while the stock price decreased of approximately 4%. This also lead to a very high sharpe and colmar ratio.
