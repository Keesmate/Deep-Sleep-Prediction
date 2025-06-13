#load data_1.csv
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns


# load data
df = pd.read_csv('Data_1.csv')

# Convert the 'Date' column to datetime and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values('Date', inplace=True)

# Step 1: add calendar features for seasonality screening
df['month'] = df['Date'].dt.month
df['dow']   = df['Date'].dt.dayofweek

#plot each column over time in a separate plot

"""def plot_columns_over_time(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = len(numeric_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=(num_cols + 2) // 3, ncols=3, figsize=(15, 5 * ((num_cols + 2) // 3)))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, col in enumerate(numeric_cols):
        axes[i].plot(df['Date'], df[col], label=col)
        axes[i].set_title(col)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(col)
        axes[i].legend()
        axes[i].grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
plot_columns_over_time(df)

# Step 1 (cont.): boxplots by month and day-of-week for each numeric column
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='month', y=col, data=df)
    plt.title(f"{col} by Month")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x='dow', y=col, data=df)
    plt.title(f"{col} by Day of Week")
    plt.show()

"""
# Step 2: seasonal decomposition for each numeric column (weekly cycle), select strong seasonal vars, and add seasonal components
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_indexed = df.set_index('Date')
strengths = []
threshold = 0.1
selected = []
for col in numeric_cols:
    # decompose weekly cycle
    result = seasonal_decompose(df_indexed[col].dropna(), period=7, model='additive')
    result.plot()
    plt.suptitle(f"Seasonal Decompose of {col}", y=1.02)
    plt.show()
    # compute strength
    var_seasonal = result.seasonal.var()
    var_resid = result.resid.var()
    strength = var_seasonal / (var_seasonal + var_resid)
    strengths.append({'variable': col, 'strength': strength})
    # if strong seasonal, add its seasonal component to features
    if strength >= threshold:
        df_indexed[f"{col}_seasonal_weekly"] = result.seasonal
        selected.append(col)

# summarize seasonality strength
strength_df = pd.DataFrame(strengths).sort_values('strength', ascending=False)
print("Seasonality Strength (weekly):")
print(strength_df)
# reset index to bring Date back as column
df = df_indexed.reset_index()
df = df.drop(columns=['dow_seasonal_weekly'])
df = df.drop(columns=['dow'])
df = df.drop(columns=['month'])
# save enriched dataset with seasonal features
df.to_csv('Data_1_seasonal_features.csv', index=False)
print(f"Selected seasonal variables (strength >= {threshold}): {selected}")
