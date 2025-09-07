import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error

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

# Drop or exclude the original time columns now that we have numeric features
df.drop(['Date',], axis=1, inplace=True)


########## Feature curation 1 ##########
df.drop(columns=['Pressure_Max', 'MaxTemp_C', 'Pressure_Avg', 'SunriseCos', 'Pressure_Min', 'SunsetSin', 'SunriseSin', 'Breathing_Sleep_Quality'], inplace=True, errors='ignore')


# ################# 2. Data Splitting ##############50##

# Define indices for split (70/15/15 split for 100 days of data)
n = len(df)
train_size = int(0.70 * n)     
val_size   = int(0.15 * n)      
test_size  = n - train_size - val_size  

# Split the dataframe into train, val, test segments
train_data = df.iloc[:train_size].copy()
val_data   = df.iloc[train_size:train_size+val_size].copy()
test_data  = df.iloc[train_size+val_size:].copy()

# Verify the split sizes (optional)
print(len(train_data), len(val_data), len(test_data))  

# Scale the feature columns using training data statistics
feature_cols = [col for col in train_data.columns if col != 'DeepSleep']  # all columns except the target
target_col = 'DeepSleep'
scaler = StandardScaler()

# Fit scaler on training data and transform all splits
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols]   = scaler.transform(val_data[feature_cols])
test_data[feature_cols]  = scaler.transform(test_data[feature_cols])

# Now train_data, val_data, test_data are scaled (each is a DataFrame).



################# 3. Modeling with TCN #################
EPOCHS = 350
SEQ_LENGTH = 10  
BATCH_SIZE = 8  
num_channels = [32, 32]
kernel = 3
dropout_size = 0.3

 
class SleepDataset(Dataset):
    def __init__(self, data_df, seq_length=SEQ_LENGTH):
        self.seq_length = seq_length
        # Split into feature array X and target array y
        self.X = data_df[feature_cols].values  # features (already scaled)
        self.y = data_df[target_col].values    # targets (deep sleep)
    
    def __len__(self):
        # Number of sequences we can generate
        return len(self.X) - self.seq_length + 1
    
    def __getitem__(self, idx):
        # Take seq_length consecutive days starting at idx
        x_seq = self.X[idx : idx + self.seq_length]
        y_target = self.y[idx + self.seq_length - 1]  # target is deep sleep on the last day of the sequence
        
        # Convert to torch tensors
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)
        # Reshape x_seq to (features, seq_length) for TCN (channels-first format)
        x_seq = x_seq.T  # originally (seq_length, features) -> transpose to (features, seq_length)
        return x_seq, y_target

# Create dataset instances for train, val, test
train_dataset = SleepDataset(train_data, seq_length=SEQ_LENGTH)
val_dataset   = SleepDataset(val_data, seq_length=SEQ_LENGTH)
test_dataset  = SleepDataset(test_data, seq_length=SEQ_LENGTH)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define a custom layer to remove padding on the right (for causal conv)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        # If padding added on both sides, remove padding on the end (right side)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

# Define a Temporal Block (two conv layers + residual connection)
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        # First 1D convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)            # remove right padding to maintain causality
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)
        # Second 1D convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)
        # Package the two conv layers into a Sequential for convenience
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                 self.conv2, self.chomp2, self.relu2, self.drop2)
        # Residual connection: 1x1 conv to match channels if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                           if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        # Add residual (skip connection)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

# Define the full TCN network
class TCN(nn.Module):
    def __init__(self, input_channels, output_size, num_channels_list, kernel_size, dropout):
        """
        input_channels: number of input features (channels)
        output_size: dimension of output (for regression = 1)
        num_channels_list: list with number of filters in each TemporalBlock
        kernel_size: convolution kernel size
        dropout: dropout rate
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels_list)
        # Build a sequence of TemporalBlocks
        for i in range(num_levels):
            in_ch = input_channels if i == 0 else num_channels_list[i-1]
            out_ch = num_channels_list[i]
            # Exponential dilation (1, 2, 4, ... for successive layers)
            dilation = 2 ** i  
            # Padding such that output length = input length (causal padding on left)
            padding = (kernel_size - 1) * dilation  
            # Add the TemporalBlock
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        # Final linear layer to map to desired output size
        self.fc = nn.Linear(num_channels_list[-1], output_size)
    
    def forward(self, x):
        """
        x shape: (batch_size, input_channels, seq_length)
        """
        # Pass through TCN layers
        y = self.tcn(x)   # shape: (batch, out_channels, seq_length) after last layer (same length due to padding/chomp)
        # Take the output at the last time step
        last_step = y[:, :, -1]  # shape: (batch, out_channels)
        out = self.fc(last_step) # shape: (batch, output_size)
        return out


# Initialize the TCN model
input_channels = len(feature_cols)  # number of input features
model = TCN(input_channels, output_size=1, num_channels_list=num_channels, kernel_size=kernel, dropout=dropout_size)

# Move model to GPU if available (optional)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

### loss plot lists
train_loss_list = []
val_loss_list = []
test_mae_list = []


for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    # Training loop
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)              # forward pass
        loss = criterion(output.squeeze(), batch_y)  # compute MSE loss 
        loss.backward()                      # backpropagation
        optimizer.step()                     # update parameters
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation loop (to monitor performance on unseen data for tuning)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze()
            val_loss += criterion(preds, batch_y).item()
    val_loss /= len(val_loader)
    
    # Print epoch metrics
    print(f"Epoch {epoch}: Train MSE = {train_loss:.4f}, Val MSE = {val_loss:.4f}")
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # --- Track test MAE per epoch ---
    model.eval()
    epoch_test_preds = []
    epoch_test_actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).squeeze()
            epoch_test_preds.append(preds.cpu().numpy())
            epoch_test_actuals.append(batch_y.cpu().numpy())

    epoch_test_preds = np.concatenate(epoch_test_preds)
    epoch_test_actuals = np.concatenate(epoch_test_actuals)
    epoch_test_mae = np.mean(np.abs(epoch_test_preds - epoch_test_actuals))
    test_mae_list.append(epoch_test_mae)

    
    # (Optional) Early stopping: you could stop if val_loss doesn't improve for several epochs


model.eval()
with torch.no_grad():
    all_preds = []
    all_actuals = []
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        preds = model(batch_X).squeeze()        # model's predictions
        all_preds.append(preds.cpu().numpy())   # collect predictions
        all_actuals.append(batch_y.numpy())     # collect true values

# Concatenate all batches
all_preds = np.concatenate(all_preds)
all_actuals = np.concatenate(all_actuals)

# Compute MSE and MAE
test_mse = np.mean((all_preds - all_actuals) ** 2)
test_mae = np.mean(np.abs(all_preds - all_actuals))
print(f"Test MSE: {test_mse:.3f}, Test MAE: {test_mae:.3f}")





# ############ print actual vs predicted ############
# # Create x-axis (time steps)
# x = np.arange(len(all_actuals))
# plt.figure(figsize=(14, 6))

# # Plot actual and predicted lines
# plt.plot(x, all_actuals[:len(x)], label='Actual', color='blue', linewidth=2)
# plt.plot(x, all_preds[:len(x)], label='Predicted', color='orange', linestyle='--', linewidth=2)

# # Labels and title
# plt.title('Deep Sleep Predictions vs Actual - Test Set')
# plt.xlabel('Time Step (Day)')
# plt.ylabel('Deep Sleep (minutes)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Show or save the plot
# plt.show()
# # plt.savefig('deep_sleep_prediction_plot.png')  # optionally save



def get_predictions(dataset, dataloader, label="Set", start_index=0):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_X)

            output_np = output.view(-1).cpu().numpy()
            target_np = batch_y.view(-1).cpu().numpy()

            preds.append(output_np)
            actuals.append(target_np)

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    mse = np.mean((preds - actuals) ** 2)
    mae = np.mean(np.abs(preds - actuals))
    print(f"{label} MSE: {mse:.3f}, MAE: {mae:.3f}")

    return preds, actuals, start_index + SEQ_LENGTH - 1  # adjust for sequence offset


train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_preds, train_actuals, train_start = get_predictions(train_dataset, train_eval_loader, "Train", start_index=0)
val_preds, val_actuals, val_start = get_predictions(val_dataset, val_loader, "Validation", start_index=len(train_data))
test_preds, test_actuals, test_start = get_predictions(test_dataset, test_loader, "Test", start_index=len(train_data) + len(val_data))


# # Plot function

# def plot_predictions(preds, actuals, title):
#     x = np.arange(len(actuals))
#     plt.figure(figsize=(14, 6))
#     plt.plot(x, actuals, label='Actual', color='blue', linewidth=2)
#     plt.plot(x, preds, label='Predicted', color='orange', linestyle='--', linewidth=2)
#     plt.title(title)
#     plt.xlabel("Time Step (Day)")
#     plt.ylabel("Deep Sleep (minutes)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Plot them all
# plot_predictions(train_preds, train_actuals, "Train Set: Deep Sleep Predictions vs Actual")
# plot_predictions(val_preds, val_actuals, "Validation Set: Deep Sleep Predictions vs Actual")
# plot_predictions(test_preds, test_actuals, "Test Set: Deep Sleep Predictions vs Actual")

def plot_all_predictions_with_correct_alignment(train_preds, train_actuals, train_start,
                                                val_preds, val_actuals, val_start,
                                                test_preds, test_actuals, test_start):
    plt.figure(figsize=(14, 6))

    # Time axes
    train_x = np.arange(train_start, train_start + len(train_preds)) + 1
    val_x = np.arange(val_start, val_start + len(val_preds)) + 1
    test_x = np.arange(test_start, test_start + len(test_preds)) + 1

    # Actuals
    plt.plot(train_x, train_actuals, label='Train Actual', color='blue', linewidth=2)
    plt.plot(val_x, val_actuals, label='Validation Actual', color='green', linewidth=2)
    plt.plot(test_x, test_actuals, label='Test Actual', color='red', linewidth=2)

    # Predictions
    plt.plot(train_x, train_preds, label='Train Predicted', color='lightblue', linestyle='--')
    plt.plot(val_x, val_preds, label='Validation Predicted', color='lightgreen', linestyle='--')
    plt.plot(test_x, test_preds, label='Test Predicted', color='lightcoral', linestyle='--')

    plt.title("Deep Sleep Predictions vs Actual")
    plt.xlabel("Day")
    plt.ylabel("Deep Sleep (mins)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_all_predictions_with_correct_alignment(
    train_preds, train_actuals, train_start,
    val_preds, val_actuals, val_start,
    test_preds, test_actuals, test_start
)


# --------------- MAE Comparison Plot ----------------

# Collect the final MAEs
train_mae = np.mean(np.abs(train_preds - train_actuals))
val_mae   = np.mean(np.abs(val_preds - val_actuals))
test_mae  = np.mean(np.abs(test_preds - test_actuals))

# Bar chart
mae_values = [train_mae, val_mae, test_mae]
labels = ['Train', 'Validation', 'Test']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, mae_values, color=colors)

# Annotate each bar with the MAE value
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.title("MAE Comparison Across Train/Val/Test Sets")
plt.ylabel("Mean Absolute Error")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Print prediction indices for each set
train_indices = list(range(train_start, train_start + len(train_preds)))
val_indices = list(range(val_start, val_start + len(val_preds)))
test_indices = list(range(test_start, test_start + len(test_preds)))

print("Train Prediction Indices:", train_indices)
print("Validation Prediction Indices:", val_indices)
print("Test Prediction Indices:", test_indices)