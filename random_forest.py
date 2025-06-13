import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
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
df = pd.read_csv('Data_1.csv')
df.rename(columns={'Deep sleep (mins)': 'DeepSleep'}, inplace=True)

# drop GMM columns
cols_to_drop = [col for col in df.columns if col.startswith("GMM")]
df = df.drop(columns=cols_to_drop)

print(df.columns)


# ################# 2. Data Splitting #################

# Define indices for split (70/15/15 split for 100 days of data)
n = len(df)
train_size = int(0.70 * n)      # 70 days
val_size   = int(0.15 * n)      # 15 days
test_size  = n - train_size - val_size  # remaining 15 days

# Split the dataframe into train, val, test segments
train_data = df.iloc[:train_size].copy()
val_data   = df.iloc[train_size:train_size+val_size].copy()
test_data  = df.iloc[train_size+val_size:].copy()

# Verify the split sizes (optional)
print(len(train_data), len(val_data), len(test_data))  # Expect 70, 15, 15

# Scale the feature columns using training data statistics
feature_cols = [col for col in train_data.columns if col != 'DeepSleep']  # all columns except the target
target_col = 'DeepSleep'
scaler = StandardScaler()

# Fit scaler on training data and transform all splits
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols]   = scaler.transform(val_data[feature_cols])
test_data[feature_cols]  = scaler.transform(test_data[feature_cols])