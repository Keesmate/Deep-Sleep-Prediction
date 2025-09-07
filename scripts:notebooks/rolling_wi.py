import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

"""

Creates rolling windows for all columns in the dataset from 1-10 days.

"""

df = pd.read_csv('Data_1.csv')



# Convert the 'Date' column to datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values('Date', inplace=True)
# set Date as index for calendar-based rolling windows
df = df.set_index('Date')


"""

Use this before you do feature engineering so we have the same way of indexing the date.

"""



# delete all columns that start with 'GMM'
df = df.loc[:, ~df.columns.str.startswith('GMM')]
# delete all columns that start with 'Unnamed'
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

# Create rolling-window mean features for windows of length 1 to 7 days
window_sizes = range(1, 11)
# use all numeric columns except the target 'DeepSleep' to generate rolling windows
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                if col != 'Deep sleep (mins)']
for w in window_sizes:
    for col in numeric_cols:
        df[f"{col}_roll{w}_mean"] = df[col].rolling(f"{w}D", min_periods=1).mean()
        df[f"{col}_roll{w}_std"] = df[col].rolling(f"{w}D", min_periods=2).std()

# remove 1 day std variables
cols_to_drop = df.columns[df.columns.str.endswith("roll1_std")]
print("Dropping columns:\n", cols_to_drop.tolist())
df = df.drop(columns=cols_to_drop)

# Fill NaNs in all *_std columns with 0
std_cols = df.columns[df.columns.str.endswith('_std')]
df[std_cols] = df[std_cols].fillna(0)

# reset index to bring Date back as a column
df = df.reset_index()
#save the sliding windows features to a new csv file
df.to_csv('sliding_windows_features.csv')

#print all column names
def print_column_names(df):
    for col in df.columns:
        print(col)

print_column_names(df)
