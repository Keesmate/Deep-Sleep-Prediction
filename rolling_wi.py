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

df = pd.read_csv('/Users/noah/PycharmProjects/QuantifedSelf/Data_1.csv')



# Convert the 'Date' column to datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values('Date', inplace=True)
# set Date as index for calendar-based rolling windows
df = df.set_index('Date')


"""

Use this before you do feature engineering so we have the same way indexing the date.

"""



# delete all columns that start with 'GMM'
df = df.loc[:, ~df.columns.str.startswith('GMM')]
# delete all columns that start with 'Unnamed'
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

# Create rolling-window mean features for windows of length 1 to 7 days
window_sizes = range(1, 11)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for w in window_sizes:
    for col in numeric_cols:
        df[f"{col}_roll{w}_mean"] = df[col].rolling(f"{w}D", min_periods=1).mean()
        df[f"{col}_roll{w}_std"] = df[col].rolling(f"{w}D", min_periods=1).std()


# reset index to bring Date back as a column
df = df.reset_index()
#save the sliding windows features to a new csv file
df.to_csv('/Users/noah/PycharmProjects/QuantifedSelf/sliding_windows_features.csv', index=False)

#print all column names
def print_column_names(df):
    for col in df.columns:
        print(col)

print_column_names(df)
