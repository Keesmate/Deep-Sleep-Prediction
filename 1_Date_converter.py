import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# load data
df = pd.read_csv('final_cleaned_data.csv')


################# 1. Data Preparation #################
# Convert the 'Date' column to datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Handle time-of-day features by converting to numeric and creating cyclical features
# Convert times to minutes since midnight
df['BedTimeMinutes'] = pd.to_datetime(df['Bed-time'], format='%H:%M %p').dt.hour * 60 + \
                       pd.to_datetime(df['Bed-time'], format='%H:%M %p').dt.minute

df['SunsetMinutes'] = pd.to_datetime(df['Sunset'], format='%H:%M').dt.hour * 60 + \
                      pd.to_datetime(df['Sunset'], format='%H:%M').dt.minute

df['WakeTimeMinutes'] = pd.to_datetime(df['Wakeup-time'], format='%H:%M %p').dt.hour * 60 + \
                       pd.to_datetime(df['Wakeup-time'], format='%H:%M %p').dt.minute

df['SunriseMinutes'] = pd.to_datetime(df['Sunrise'], format='%H:%M').dt.hour * 60 + \
                      pd.to_datetime(df['Sunrise'], format='%H:%M').dt.minute


# Create cyclical features for bed time + wake time (24h = 1440 minutes period)
df['BedTimeSin'] = np.sin(2 * np.pi * df['BedTimeMinutes'] / 1440)
df['BedTimeCos'] = np.cos(2 * np.pi * df['BedTimeMinutes'] / 1440)
df['WakeTimeSin'] = np.sin(2 * np.pi * df['WakeTimeMinutes'] / 1440)
df['WakeTimeCos'] = np.cos(2 * np.pi * df['WakeTimeMinutes'] / 1440)

# Create cyclical features for sunset + sunrise (24h = 1440 minutes period)
df['SunsetSin'] = np.sin(2 * np.pi * df['SunsetMinutes'] / 1440)
df['SunsetCos'] = np.cos(2 * np.pi * df['SunsetMinutes'] / 1440)
df['SunriseSin'] = np.sin(2 * np.pi * df['SunriseMinutes'] / 1440)
df['SunriseCos'] = np.cos(2 * np.pi * df['SunriseMinutes'] / 1440)


print(df["BedTimeSin"].head())

# (Optional) One could create cyclical features for other daily cycles (e.g., if using day-of-week)

# Drop or exclude the original time columns now that we have numeric features
df.drop(['Date', 'Bed-time', 'Wakeup-time', 'BedTimeMinutes', 'WakeTimeMinutes', 
         'Sunrise', 'Sunset', 'SunriseMinutes', 'SunsetMinutes'], axis=1, inplace=True)

df.to_csv('Data_1.csv', index=False)