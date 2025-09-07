from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


# ############ ARIMA #############
# # Load data
# df = pd.read_csv('cleaned_data.csv', parse_dates=['Date'])
# df.sort_values('Date', inplace=True)

# # Set index for ARIMA
# ts = df['Deep sleep (mins)']
# ts.index = df['Date']

# # Fit ARIMA (simple model, can be tuned)
# model = ARIMA(ts, order=(1, 1, 1))
# model_fit = model.fit()

# # Forecast next 10 days
# forecast = model_fit.forecast(steps=10)
# print(forecast)

########### lag correlation ###########
# Example: load your data
df = pd.read_csv('Data_1.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
ts = df['DeepSleep']

ts.plot(figsize=(12, 4), title="Time Series Plot")
plt.xlabel("Day")
plt.ylabel("Deep Sleep (mins)")
plt.grid(True)
plt.show()

plot_acf(ts, lags=30)
plt.title("Autocorrelation Plot")
plt.grid(True)
plt.show()

