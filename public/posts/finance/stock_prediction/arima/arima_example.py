import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Download stock data
ticker = "MSFT"
start_date = "2018-01-01"
end_date = "2023-06-23"
data = yf.download(ticker, start=start_date, end=end_date)

# Prepare the data
ts = data['Close']

# Check for stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

test_stationarity(ts)

# If non-stationary, difference the series
ts_diff = ts.diff().dropna()
test_stationarity(ts_diff)

# Fit ARIMA model
model = ARIMA(ts_diff, order=(1,1,1))
results = model.fit()
print(results.summary())

# Forecast
forecast = results.forecast(steps=30)
# help(forecast)
print(forecast)

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(ts.index[-100:], ts.values[-100:], label='Observed')
plt.plot(forecast.index, forecast.values, color='r', label='Forecast')
# plt.fill_between(forecast.index, 
#                  forecast.iloc[:, 0], 
#                  forecast.iloc[:, 1], 
#                  color='pink', alpha=0.3)
plt.title(f'{ticker} Stock Price Prediction')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(ts.diff().dropna()[-30:], forecast)
print(f'Mean Squared Error: {mse}')