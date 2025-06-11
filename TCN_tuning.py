import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Load and prepare the data
df = pd.read_csv("Data_1.csv")
df.rename(columns={'Deep sleep (mins)': 'DeepSleep'}, inplace=True)
target_col = 'DeepSleep'
feature_cols = [col for col in df.columns if col != target_col]

# Split
n = len(df)
train_size = int(0.70 * n)
val_size   = int(0.15 * n)
test_size  = n - train_size - val_size
train_data = df.iloc[:train_size].copy()
val_data   = df.iloc[train_size:train_size+val_size].copy()
test_data  = df.iloc[train_size+val_size:].copy()

# Scale
scaler = StandardScaler()
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols] = scaler.transform(val_data[feature_cols])
test_data[feature_cols] = scaler.transform(test_data[feature_cols])

# Dataset
class SleepDataset(Dataset):
    def __init__(self, data_df, seq_length):
        self.seq_length = seq_length
        self.X = data_df[feature_cols].values
        self.y = data_df[target_col].values

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length - 1]
        x_seq = torch.tensor(x_seq, dtype=torch.float32).T
        y_target = torch.tensor(y_target, dtype=torch.float32)
        return x_seq, y_target

# TCN components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class TCN(nn.Module):
    def __init__(self, input_channels, output_size, num_channels_list, kernel_size=3, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(num_channels_list)):
            in_ch = input_channels if i == 0 else num_channels_list[i-1]
            out_ch = num_channels_list[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels_list[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        return self.fc(y[:, :, -1])

# Grid to search
param_grid = {
    'SEQ_LENGTH': [5, 7, 10, 14],
    'num_channels_list': [
        [16, 16],
        [32, 32],
        [32, 64],
        [64, 64, 64]
    ],
    'kernel_size': [3, 5, 7],
    'dropout': [0.1, 0.2, 0.3, 0.5],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [8, 16, 32],
    'EPOCHS': [50, 100, 200, 500]
}

# Track results
results = []
best_mae = float('inf')
best_config = None

# Run grid search
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for config_values in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), config_values))
    try:
        train_dataset = SleepDataset(train_data, config['SEQ_LENGTH'])
        val_dataset = SleepDataset(val_data, config['SEQ_LENGTH'])
        test_dataset = SleepDataset(test_data, config['SEQ_LENGTH'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = TCN(
            input_channels=len(feature_cols),
            output_size=1,
            num_channels_list=config['num_channels_list'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()

        for epoch in range(config['EPOCHS']):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                pred = model(batch_X).squeeze().cpu().numpy()
                preds.append(pred)
                targets.append(batch_y.numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        mae = np.mean(np.abs(preds - targets))
        results.append((config, mae))
        if mae < best_mae:
            best_mae = mae
            best_config = config
        print(f"Test MAE: {mae:.2f} for config: {config}")
    except Exception as e:
        print(f"Failed for config {config} with error: {e}")

# Save results
results_df = pd.DataFrame(results, columns=["Config", "Test_MAE"])
results_df.to_csv("tcn_grid_search_results.csv", index=False)
print("\nBest Config:\n", best_config)
print("Best MAE:", best_mae)
