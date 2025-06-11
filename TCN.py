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

df.to_csv('processed_data.csv', index=False)

# At this point, `df` contains all numeric features (e.g., Steps, HeartRate, TotalSleep, SunsetMinutes, 
# BedTimeSin, BedTimeCos, etc.) and the target 'DeepSleep'.
# We will scale these features next (after splitting the data to avoid data leakage).




# ################# 2. Data Splitting #################

# # Define indices for split (70/15/15 split for 100 days of data)
# n = len(df)
# train_size = int(0.70 * n)      # 70 days
# val_size   = int(0.15 * n)      # 15 days
# test_size  = n - train_size - val_size  # remaining 15 days

# # Split the dataframe into train, val, test segments
# train_data = df.iloc[:train_size].copy()
# val_data   = df.iloc[train_size:train_size+val_size].copy()
# test_data  = df.iloc[train_size+val_size:].copy()

# # Verify the split sizes (optional)
# print(len(train_data), len(val_data), len(test_data))  # Expect 70, 15, 15

# # Scale the feature columns using training data statistics
# feature_cols = [col for col in train_data.columns if col != 'DeepSleep']  # all columns except the target
# target_col = 'DeepSleep'
# scaler = StandardScaler()

# # Fit scaler on training data and transform all splits
# scaler.fit(train_data[feature_cols])
# train_data[feature_cols] = scaler.transform(train_data[feature_cols])
# val_data[feature_cols]   = scaler.transform(val_data[feature_cols])
# test_data[feature_cols]  = scaler.transform(test_data[feature_cols])

# # Now train_data, val_data, test_data are scaled (each is a DataFrame).



# ################# 3. Modeling with TCN #################
# SEQ_LENGTH = 7  # for example, use 7-day sequences (this can be tuned)

# class SleepDataset(Dataset):
#     def __init__(self, data_df, seq_length=SEQ_LENGTH):
#         self.seq_length = seq_length
#         # Split into feature array X and target array y
#         self.X = data_df[feature_cols].values  # features (already scaled)
#         self.y = data_df[target_col].values    # targets (deep sleep)
    
#     def __len__(self):
#         # Number of sequences we can generate
#         return len(self.X) - self.seq_length + 1
    
#     def __getitem__(self, idx):
#         # Take seq_length consecutive days starting at idx
#         x_seq = self.X[idx : idx + self.seq_length]
#         y_target = self.y[idx + self.seq_length - 1]  # target is deep sleep on the last day of the sequence
        
#         # Convert to torch tensors
#         x_seq = torch.tensor(x_seq, dtype=torch.float32)
#         y_target = torch.tensor(y_target, dtype=torch.float32)
#         # Reshape x_seq to (features, seq_length) for TCN (channels-first format)
#         x_seq = x_seq.T  # originally (seq_length, features) -> transpose to (features, seq_length)
#         return x_seq, y_target

# # Create dataset instances for train, val, test
# train_dataset = SleepDataset(train_data, seq_length=SEQ_LENGTH)
# val_dataset   = SleepDataset(val_data, seq_length=SEQ_LENGTH)
# test_dataset  = SleepDataset(test_data, seq_length=SEQ_LENGTH)

# # Create DataLoader for batching
# BATCH_SIZE = 16  # example batch size, can be tuned
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# # Define a custom layer to remove padding on the right (for causal conv)
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super().__init__()
#         self.chomp_size = chomp_size
#     def forward(self, x):
#         # If padding added on both sides, remove padding on the end (right side)
#         return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

# # Define a Temporal Block (two conv layers + residual connection)
# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.0):
#         super().__init__()
#         # First 1D convolution
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
#                                 stride=stride, padding=padding, dilation=dilation)
#         self.chomp1 = Chomp1d(padding)            # remove right padding to maintain causality
#         self.relu1  = nn.ReLU()
#         self.drop1  = nn.Dropout(dropout)
#         # Second 1D convolution
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
#                                 stride=stride, padding=padding, dilation=dilation)
#         self.chomp2 = Chomp1d(padding)
#         self.relu2  = nn.ReLU()
#         self.drop2  = nn.Dropout(dropout)
#         # Package the two conv layers into a Sequential for convenience
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
#                                  self.conv2, self.chomp2, self.relu2, self.drop2)
#         # Residual connection: 1x1 conv to match channels if needed
#         self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
#                            if in_channels != out_channels else None
#         self.relu_out = nn.ReLU()
    
#     def forward(self, x):
#         out = self.net(x)
#         # Add residual (skip connection)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu_out(out + res)

# # Define the full TCN network
# class TCN(nn.Module):
#     def __init__(self, input_channels, output_size, num_channels_list, kernel_size=3, dropout=0.0):
#         """
#         input_channels: number of input features (channels)
#         output_size: dimension of output (for regression = 1)
#         num_channels_list: list with number of filters in each TemporalBlock
#         kernel_size: convolution kernel size
#         dropout: dropout rate
#         """
#         super().__init__()
#         layers = []
#         num_levels = len(num_channels_list)
#         # Build a sequence of TemporalBlocks
#         for i in range(num_levels):
#             in_ch = input_channels if i == 0 else num_channels_list[i-1]
#             out_ch = num_channels_list[i]
#             # Exponential dilation (1, 2, 4, ... for successive layers)
#             dilation = 2 ** i  
#             # Padding such that output length = input length (causal padding on left)
#             padding = (kernel_size - 1) * dilation  
#             # Add the TemporalBlock
#             layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
#                                         dilation=dilation, padding=padding, dropout=dropout))
#         self.tcn = nn.Sequential(*layers)
#         # Final linear layer to map to desired output size
#         self.fc = nn.Linear(num_channels_list[-1], output_size)
    
#     def forward(self, x):
#         """
#         x shape: (batch_size, input_channels, seq_length)
#         """
#         # Pass through TCN layers
#         y = self.tcn(x)   # shape: (batch, out_channels, seq_length) after last layer (same length due to padding/chomp)
#         # Take the output at the last time step
#         last_step = y[:, :, -1]  # shape: (batch, out_channels)
#         out = self.fc(last_step) # shape: (batch, output_size)
#         return out


# # Initialize the TCN model
# input_channels = len(feature_cols)  # number of input features
# model = TCN(input_channels, output_size=1, num_channels_list=[16, 16], kernel_size=3, dropout=0.2)

# # Move model to GPU if available (optional)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)


# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # learning rate can be tuned

# EPOCHS = 50  # for example, train for 50 epochs
# for epoch in range(1, EPOCHS+1):
#     model.train()
#     train_loss = 0.0
#     # Training loop
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         output = model(batch_X)              # forward pass
#         loss = criterion(output.squeeze(), batch_y)  # compute MSE loss 
#         loss.backward()                      # backpropagation
#         optimizer.step()                     # update parameters
#         train_loss += loss.item()
#     train_loss /= len(train_loader)
    
#     # Validation loop (to monitor performance on unseen data for tuning)
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_X, batch_y in val_loader:
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#             preds = model(batch_X).squeeze()
#             val_loss += criterion(preds, batch_y).item()
#     val_loss /= len(val_loader)
    
#     # Print epoch metrics
#     print(f"Epoch {epoch}: Train MSE = {train_loss:.4f}, Val MSE = {val_loss:.4f}")
    
#     # (Optional) Early stopping: you could stop if val_loss doesn't improve for several epochs


# model.eval()
# with torch.no_grad():
#     all_preds = []
#     all_actuals = []
#     for batch_X, batch_y in test_loader:
#         batch_X = batch_X.to(device)
#         preds = model(batch_X).squeeze()        # model's predictions
#         all_preds.append(preds.cpu().numpy())   # collect predictions
#         all_actuals.append(batch_y.numpy())     # collect true values

# # Concatenate all batches
# all_preds = np.concatenate(all_preds)
# all_actuals = np.concatenate(all_actuals)

# # Compute MSE and MAE
# test_mse = np.mean((all_preds - all_actuals) ** 2)
# test_mae = np.mean(np.abs(all_preds - all_actuals))
# print(f"Test MSE: {test_mse:.3f}, Test MAE: {test_mae:.3f}")
