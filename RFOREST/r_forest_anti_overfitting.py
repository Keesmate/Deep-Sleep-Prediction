import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import optuna
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

# Load and prepare data
df = pd.read_csv('/Data_1.csv')
df.rename(columns={'Deep sleep (mins)': 'DeepSleep'}, inplace=True)

# Drop GMM columns
cols_to_drop = [col for col in df.columns if col.startswith("GMM")]
df = df.drop(columns=cols_to_drop)

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(df.columns) - 2}")  # excluding target and date

# ################# IMPROVED DATA SPLITTING FOR TIME SERIES #################
# Use time-aware splitting - no random shuffling for temporal data
n = len(df)
train_size = int(0.70 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size

train_data = df.iloc[:train_size].copy()
val_data = df.iloc[train_size:train_size + val_size].copy()
test_data = df.iloc[train_size + val_size:].copy()

print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Feature preparation
feature_cols = [col for col in train_data.columns if col not in ['DeepSleep', 'Date']]
target_col = 'DeepSleep'

# Standardization
scaler = StandardScaler()
scaler.fit(train_data[feature_cols])
train_data[feature_cols] = scaler.transform(train_data[feature_cols])
val_data[feature_cols] = scaler.transform(val_data[feature_cols])
test_data[feature_cols] = scaler.transform(test_data[feature_cols])


# ################# IMPROVED FEATURE SELECTION #################
def robust_feature_selection(X_train, y_train, max_features=12):
    """
    More conservative feature selection to reduce overfitting
    """
    # Use a simpler model for feature selection
    base_selector = RandomForestRegressor(
        n_estimators=50,  # Much smaller for feature selection
        max_depth=3,  # Shallow trees to avoid overfitting
        random_state=42,
        n_jobs=-1
    )

    # Use RFECV with time series CV for more robust selection
    tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits for small dataset

    selector = RFECV(
        estimator=base_selector,
        step=1,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        min_features_to_select=min(5, len(feature_cols) // 3)  # At least 5 features
    )

    selector.fit(X_train, y_train)

    # If too many features selected, use importance-based selection to limit
    if selector.n_features_ > max_features:
        print(f"RFECV selected {selector.n_features_} features, reducing to {max_features}")

        # Fit a model to get feature importances
        temp_model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        selected_idx = [i for i in range(len(feature_cols)) if selector.support_[i]]
        temp_features = [feature_cols[i] for i in selected_idx]
        temp_model.fit(X_train[temp_features], y_train)

        # Get top features by importance
        importances = temp_model.feature_importances_
        top_idx = np.argsort(importances)[-max_features:]

        selected_features = [temp_features[i] for i in top_idx]
    else:
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]

    print(f"Selected {len(selected_features)} features: {selected_features}")
    return selected_features


# ################# CONSERVATIVE HYPERPARAMETER TUNING #################
def objective(trial):
    # Much more conservative hyperparameter ranges
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),  # Reduced range
        "max_depth": trial.suggest_int("max_depth", 3, 8),  # Shallower trees
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 15),  # Higher minimum
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 8),  # Higher minimum
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 0.001, 0.1, log=True),  # More aggressive pruning
        "max_samples": trial.suggest_float("max_samples", 0.7, 0.95)  # Bootstrap sampling
    }

    # Use TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=3)  # 3 splits for 70 samples

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

    cv_scores = cross_val_score(
        model,
        train_data[selected_features],
        train_data[target_col],
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    return -cv_scores.mean()


# Perform feature selection
selected_features = robust_feature_selection(
    train_data[feature_cols],
    train_data[target_col],
    max_features=12  # Conservative feature count
)

# Hyperparameter optimization with early stopping
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500, n_jobs=-1)  # Fewer trials to prevent overfitting

best_params = study.best_params
print("Best hyperparameters:", best_params)

# ################# REGULARIZED FINAL MODEL #################
# Train final model with additional regularization
final_model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1,
    oob_score=True  # Enable out-of-bag scoring for additional validation
)

# Fit on training data only (not train+val) for better generalization assessment
final_model.fit(train_data[selected_features], train_data[target_col])
# Print selected features and hyperparameters for final model
print("Selected features for final model:", selected_features)
print("Selected hyperparameters for final model:", final_model.get_params())


# ################# COMPREHENSIVE EVALUATION #################
def evaluate_model(model, data, features, target, set_name):
    preds = model.predict(data[features])
    mse = mean_squared_error(data[target], preds)
    mae = mean_absolute_error(data[target], preds)

    print(f"{set_name} MSE: {mse:.4f}")
    print(f"{set_name} MAE: {mae:.4f}")
    print(f"{set_name} RMSE: {np.sqrt(mse):.4f}")

    return preds, mse, mae


# Evaluate on all sets
train_preds, train_mse, train_mae = evaluate_model(final_model, train_data, selected_features, target_col, "Train")
val_preds, val_mse, val_mae = evaluate_model(final_model, val_data, selected_features, target_col, "Val")
test_preds, test_mse, test_mae = evaluate_model(final_model, test_data, selected_features, target_col, "Test")

# Print OOB score if available
if hasattr(final_model, 'oob_score_') and final_model.oob_score_ is not None:
    print(f"OOB Score: {final_model.oob_score_:.4f}")

# ################# OVERFITTING DIAGNOSTICS (UPDATED FOR MAE) #################
print("\n=== OVERFITTING ANALYSIS ===")
# Use MAE ratios to match optimization goal
train_val_ratio_mae = val_mae / train_mae
test_val_ratio_mae = test_mae / val_mae
test_train_ratio_mae = test_mae / train_mae

print(f"Val/Train MAE ratio: {train_val_ratio_mae:.2f} (should be close to 1.0)")
print(f"Test/Val MAE ratio: {test_val_ratio_mae:.2f} (should be close to 1.0)")
print(f"Test/Train MAE ratio: {test_train_ratio_mae:.2f} (should be close to 1.0)")

if test_train_ratio_mae > 2.0:
    print("⚠️  Significant overfitting detected!")
elif test_train_ratio_mae > 1.5:
    print("⚠️  Moderate overfitting detected")
else:
    print("✅ Overfitting appears controlled")

# ################# VISUALIZATION #################
# Time series plot showing predictions over time
plt.figure(figsize=(12, 6))
train_days = range(len(train_data))
val_days = range(len(train_data), len(train_data) + len(val_data))
test_days = range(len(train_data) + len(val_data), len(train_data) + len(val_data) + len(test_data))

plt.plot(train_days, train_data[target_col].values, 'b-', label='Train Actual', alpha=0.7)
plt.plot(train_days, train_preds, 'b--', label='Train Predicted', alpha=0.7)
plt.plot(val_days, val_data[target_col].values, 'g-', label='Val Actual', alpha=0.7)
plt.plot(val_days, val_preds, 'g--', label='Val Predicted', alpha=0.7)
plt.plot(test_days, test_data[target_col].values, 'r-', label='Test Actual', alpha=0.7)
plt.plot(test_days, test_preds, 'r--', label='Test Predicted', alpha=0.7)
plt.axvline(x=len(train_data), color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=len(train_data) + len(val_data), color='gray', linestyle=':', alpha=0.5)
plt.title('Deep Sleep Predictions Over Time')
plt.xlabel('Day')
plt.ylabel('Deep Sleep (minutes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Separate plot for validation and test sets only
plt.figure(figsize=(10, 6))
# Validation period
plt.plot(val_days, val_data[target_col].values, 'g-', label='Val Actual')
plt.plot(val_days, val_preds, 'g--', label='Val Predicted')
# Test period
plt.plot(test_days, test_data[target_col].values, 'r-', label='Test Actual')
plt.plot(test_days, test_preds, 'r--', label='Test Predicted')
plt.axvline(x=len(train_data), color='gray', linestyle=':', alpha=0.5)
plt.title('Validation and Test Predictions Over Time')
plt.xlabel('Day Index')
plt.ylabel('Deep Sleep (minutes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Feature importance plot (top 10 only to reduce clutter)
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = min(10, len(selected_features))

plt.figure(figsize=(10, 6))
plt.title(f"Top {top_n} Feature Importances")
plt.barh(range(top_n), importances[indices[:top_n]])
plt.yticks(range(top_n), [selected_features[i] for i in indices[:top_n]])
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ################# ADDITIONAL VALIDATION (UPDATED FOR MAE) #################
# Learning curve analysis to detect overfitting - now using MAE to match optimization goal
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestRegressor(**best_params, random_state=42, n_jobs=-1),
    train_data[selected_features],
    train_data[target_col],
    cv=TimeSeriesSplit(n_splits=3),
    train_sizes=np.linspace(0.3, 1.0, 5),
    scoring='neg_mean_absolute_error',  # Changed from MSE to MAE
    n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), 'b-', label='Training MAE')
plt.fill_between(train_sizes, -train_scores.mean(axis=1) - train_scores.std(axis=1),
                 -train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
plt.plot(train_sizes, -val_scores.mean(axis=1), 'r-', label='Validation MAE')
plt.fill_between(train_sizes, -val_scores.mean(axis=1) - val_scores.std(axis=1),
                 -val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Absolute Error')  # Changed from MSE to MAE
plt.title('Learning Curves (MAE)')  # Updated title
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ################# MAE PROGRESSION PLOT #################
# Additional plot showing MAE progression across train/val/test
mae_values = [train_mae, val_mae, test_mae]
set_names = ['Train', 'Validation', 'Test']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 6))
bars = plt.bar(set_names, mae_values, color=colors, alpha=0.7)
plt.ylabel('Mean Absolute Error')
plt.title('MAE Comparison Across Train/Val/Test Sets')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mae_val in zip(bars, mae_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
             f'{mae_val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n=== RECOMMENDATIONS ===")
print("1. Consider collecting more data if possible")
print("2. Try simpler models (Linear Regression, Ridge) as baselines")
print("3. Implement early stopping in hyperparameter optimization")
print("4. Consider ensemble methods with different algorithms")
print("5. Add temporal features if not already present")
print("6. Try different validation strategies (blocked time series)")
print("7. Model optimized for MAE - all plots now consistent with optimization goal")