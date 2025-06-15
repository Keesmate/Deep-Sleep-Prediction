''' this script takes in the enhanced dataset and adds log transformed features, and other  enginneered features '''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random

# set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # use a fixed seed of your choice

# load data and rename target column
df = pd.read_csv('enhanced_dataset.csv')
df.rename(columns={'Deep sleep (mins)': 'DeepSleep'}, inplace=True)

################# 1. Data Preparation #################
# Convert the 'Date' column to datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Handle time-of-day features by converting to numeric and creating cyclical features
# Convert times to minutes since midnight
df['BedTimeMinutes'] = pd.to_datetime(df['Bed-time'], format='%I:%M %p').dt.hour * 60 + \
                       pd.to_datetime(df['Bed-time'], format='%I:%M %p').dt.minute

df['SunsetMinutes'] = pd.to_datetime(df['Sunset'], format='%H:%M').dt.hour * 60 + \
                      pd.to_datetime(df['Sunset'], format='%H:%M').dt.minute

df['WakeTimeMinutes'] = pd.to_datetime(df['Wakeup-time'], format='%I:%M %p').dt.hour * 60 + \
                       pd.to_datetime(df['Wakeup-time'], format='%I:%M %p').dt.minute

df['SunriseMinutes'] = pd.to_datetime(df['Sunrise'], format='%H:%M').dt.hour * 60 + \
                      pd.to_datetime(df['Sunrise'], format='%H:%M').dt.minute


# minutes after midnight
def minutes_after_sunset(bed_min, sunset_min):
    """
    Computes how many minutes bedtime is after sunset, accounting for wraparound at midnight.
    Returns a value between 0 and 1439.
    """
    if bed_min < 500:
        bed_min += 1440  # Adjust for bed time before 8:20 PM (500 minutes)
    diff = bed_min - sunset_min
    return diff

# Apply function row-wise to compute minutes after sunset
df['MinutesAfterSunset'] = df.apply(
    lambda row: minutes_after_sunset(row['BedTimeMinutes'], row['SunsetMinutes']),
    axis=1
)

############# function to calculate whether bedtime is on time, late or early ##########
# Calculate mean and standard deviation
mean_diff = df['MinutesAfterSunset'].mean()
std_diff = df['MinutesAfterSunset'].std()
print(std_diff * .5)

# Categorize as Early / On Time / Late based on 1 hour before or after average bed time
def categorize_bedtime(diff, mean, std):
    if diff < mean - (std * 0.5):
        return 'Early'
    elif diff > mean + (std * 0.5):
        return 'Late'
    else:
        return 'On Time'

df['BedtimeCategory'] = df['MinutesAfterSunset'].apply(lambda x: categorize_bedtime(x, mean_diff, std_diff))


# One-hot encode the two categorical columns
df = pd.get_dummies(df, columns=['BedtimeCategory'])

# drop deep sleep related engineered features (data leakage)
df.drop(['Deep_Sleep_Ratio'], axis=1, inplace=True)
df.drop(['Sleep_Efficiency'], axis=1, inplace=True)
df.drop(['Sleep score'], axis=1, inplace=True)
df.drop(['Sleep_Quality_Score'], axis=1, inplace=True)
df.drop(['Sleep_Battery_Interaction'], axis=1, inplace=True)
df.drop(['Sleep_Duration_Category'], axis=1, inplace=True)
df.drop(['Body Battery'], axis=1, inplace=True)

print(df.columns)


df.to_csv('Data_1.csv', index=False)