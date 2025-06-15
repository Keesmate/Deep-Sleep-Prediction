import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import optuna
from sklearn.model_selection import cross_val_score


# set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # use a fixed seed of your choice

# load data and rename target column
df = pd.read_csv('/Data_1_seasonal_features_clean.csv')
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

 # define feature columns: exclude the target and the Date column
feature_cols = [col for col in train_data.columns if col not in ['DeepSleep', 'Date']]
target_col = 'DeepSleep'
scaler = StandardScaler()

# Fit scaler on training data and transform all splits
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols]   = scaler.transform(val_data[feature_cols])
test_data[feature_cols]  = scaler.transform(test_data[feature_cols])

# ===== Feature selection and hyperparameter tuning with Optuna =====
def objective(trial):
    # suggest a threshold for feature importances
    thresh = trial.suggest_float("threshold", 0.01, 0.2)
    # perform feature selection
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        threshold=thresh
    )
    selector.fit(train_data[feature_cols], train_data[target_col])
    selected_idx = selector.get_support(indices=True)
    selected_feats = [feature_cols[i] for i in selected_idx]
    # skip trials that select no features
    if len(selected_feats) == 0:
        raise optuna.TrialPruned()
    # suggest hyperparameters for RandomForest
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 1e-1, log=True)
    }
    # evaluate with 5-fold cross-validation on train_data
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(
        model,
        train_data[selected_feats],
        train_data[target_col],
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    # return average MAE across folds
    return -cv_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=150, n_jobs=-1)
# copy best parameters to modify threshold without affecting the study's internal params
best_params = study.best_params.copy()
print("Best hyperparameters:", best_params)
# retrain final model on train+val
best_thresh = best_params.pop("threshold")
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                           threshold=best_thresh)
selector.fit(pd.concat([train_data, val_data])[feature_cols],
             pd.concat([train_data, val_data])[target_col])
final_feats = [feature_cols[i] for i in selector.get_support(indices=True)]
final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
# Fit the final model
final_model.fit(pd.concat([train_data, val_data])[final_feats],
                pd.concat([train_data, val_data])[target_col])
# Compute and print Train MSE on the training set
# Compute and print Train MSE on the training set
train_preds = final_model.predict(train_data[final_feats])
train_mse = mean_squared_error(train_data[target_col], train_preds)
print(f"Train MSE: {train_mse:.4f}")
train_mae = mean_absolute_error(train_data[target_col], train_preds)
print(f"Train MAE: {train_mae:.4f}")
# Compute and print Val MSE on the validation set
val_preds = final_model.predict(val_data[final_feats])
val_mse = mean_squared_error(val_data[target_col], val_preds)
print(f"Val MSE: {val_mse:.4f}")
val_mae = mean_absolute_error(val_data[target_col], val_preds)
print(f"Val MAE: {val_mae:.4f}")
# evaluate on test set
test_preds = final_model.predict(test_data[final_feats])
mse_test = mean_squared_error(test_data[target_col], test_preds)
print(f"Test MSE: {mse_test:.4f}")
test_mae = mean_absolute_error(test_data[target_col], test_preds)
print(f"Test MAE: {test_mae:.4f}")

# Plot actual vs predicted deep sleep on the test set
days = range(len(test_data))
plt.figure(figsize=(10, 5))
plt.plot(days, test_data[target_col].values, label='Actual', color='blue')
plt.plot(days, test_preds, label='Predicted', color='orange', linestyle='--')
plt.title('Deep Sleep Predictions vs Actual - Test Set')
plt.xlabel('Time Step (Day)')
plt.ylabel('Deep Sleep (minutes)')
plt.legend()
plt.show()

# Plot feature importances for the final selected features (horizontal bar chart)
importances = final_model.feature_importances_
# sort features by importance descending
indices = np.argsort(importances)[::-1]
sorted_feats = [final_feats[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.barh(sorted_feats, sorted_importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # highest importance at the top
plt.tight_layout()
plt.show()