#!/usr/bin/env python3
"""
Feature Engineering Script for Sleep Data Analysis
This script performs comprehensive feature engineering on sleep and health data,
adds new features, and analyzes correlations with the target variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_prepare_data(file_path='cleaned_data.csv'):
    """Load and prepare the dataset"""
    print("Loading dataset...")

    # Try different separators to find the correct one
    try:
        df = pd.read_csv(file_path, sep=',')
        print("Successfully loaded with comma separator")
    except:
        try:
            df = pd.read_csv(file_path, sep=';')
            print("Successfully loaded with semicolon separator")
        except:
            df = pd.read_csv(file_path)
            print("Successfully loaded with default separator")

    print(f"Columns found: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Check if Date column exists and convert it
    if 'Date' in df.columns:
        # Try different date formats
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            print("Date converted with format %d/%m/%Y")
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                print("Date converted with format %Y-%m-%d")
            except:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                print("Date converted with inferred format")
    else:
        print("Warning: No 'Date' column found!")
        print("Available columns:", df.columns.tolist())

    print(f"Dataset loaded successfully!")
    if 'Date' in df.columns:
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


def create_time_features(df):
    """Create time-based features"""
    print("Creating time-based features...")

    if 'Date' not in df.columns:
        print("Warning: No Date column found, skipping time features")
        return df

    # Extract time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Create weekend indicator
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Create season feature
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    print("✓ Time features created successfully")
    return df


def create_sleep_features(df):
    """Create sleep-related engineered features"""
    print("Creating sleep-related features...")

    # Check for required columns and create features if they exist
    sleep_cols = {
        'deep_sleep': None,
        'rem_sleep': None,
        'duration': None,
        'sleep_score': None,
        'restless': None,
        'hrv': None,
        'resting_hr': None
    }

    # Find matching columns (case-insensitive and flexible matching)
    for col in df.columns:
        col_lower = col.lower()
        if 'deep' in col_lower and 'sleep' in col_lower:
            sleep_cols['deep_sleep'] = col
        elif 'rem' in col_lower and 'sleep' in col_lower:
            sleep_cols['rem_sleep'] = col
        elif 'duration' in col_lower:
            sleep_cols['duration'] = col
        elif 'sleep' in col_lower and 'score' in col_lower:
            sleep_cols['sleep_score'] = col
        elif 'restless' in col_lower:
            sleep_cols['restless'] = col
        elif 'hrv' in col_lower:
            sleep_cols['hrv'] = col
        elif 'resting' in col_lower and 'heart' in col_lower:
            sleep_cols['resting_hr'] = col

    # Create features based on available columns
    if sleep_cols['deep_sleep'] and sleep_cols['rem_sleep'] and sleep_cols['duration']:
        # Sleep efficiency (Deep + REM / Total Duration)
        df['Sleep_Efficiency'] = (df[sleep_cols['deep_sleep']] + df[sleep_cols['rem_sleep']]) / df[
            sleep_cols['duration']]

        # Deep sleep ratio
        df['Deep_Sleep_Ratio'] = df[sleep_cols['deep_sleep']] / df[sleep_cols['duration']]

        # REM sleep ratio
        df['REM_Sleep_Ratio'] = df[sleep_cols['rem_sleep']] / df[sleep_cols['duration']]
        print("✓ Sleep efficiency and ratio features created")

    # Sleep quality score (custom metric)
    if sleep_cols['sleep_score'] and sleep_cols['restless'] and sleep_cols['hrv']:
        df['Sleep_Quality_Score'] = (
                df[sleep_cols['sleep_score']] * 0.4 +
                (100 - df[sleep_cols['restless']]) * 0.3 +
                df[sleep_cols['hrv']] * 0.3
        )
        print("✓ Sleep quality score created")

    # Sleep duration categories
    if sleep_cols['duration']:
        df['Sleep_Duration_Category'] = pd.cut(
            df[sleep_cols['duration']],
            bins=[0, 360, 480, 600, float('inf')],
            labels=['Short', 'Normal', 'Long', 'Very_Long']
        )
        print("✓ Sleep duration categories created")

    # Heart rate recovery (assuming lower is better during sleep)
    if sleep_cols['resting_hr']:
        df['HR_Recovery_Score'] = 100 - df[sleep_cols['resting_hr']]
        print("✓ HR recovery score created")

    return df


def create_weather_features(df):
    """Create weather-related engineered features"""
    print("Creating weather features...")

    # Check if weather columns exist
    weather_cols = [col for col in df.columns if any(weather in col.lower()
                                                     for weather in ['temp', 'humid', 'wind', 'pressure', 'precip'])]

    if weather_cols:
        # Temperature comfort index (assuming optimal sleep temp around 18-22°C)
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        if temp_cols:
            df['Temp_Comfort_Index'] = df[temp_cols].mean(axis=1).apply(
                lambda x: 100 - abs(x - 20) * 5 if not pd.isna(x) else np.nan
            )

        # Humidity comfort (optimal around 40-60%)
        humidity_cols = [col for col in df.columns if 'humid' in col.lower()]
        if humidity_cols:
            df['Humidity_Comfort_Index'] = df[humidity_cols].mean(axis=1).apply(
                lambda x: 100 - abs(x - 50) * 2 if not pd.isna(x) else np.nan
            )

        # Weather stability (low variation is better for sleep)
        if len(temp_cols) > 1:
            df['Weather_Stability'] = 100 - df[temp_cols].std(axis=1) * 10

    return df


def create_rolling_features(df, target_col='Deep sleep (mins)'):
    """Create rolling window features"""
    print("Creating rolling window features...")

    # Sort by date for rolling calculations
    df = df.sort_values('Date').reset_index(drop=True)

    # Determine window size based on data size
    window_size = min(7, len(df) // 3)  # Use smaller window if dataset is small
    min_periods = max(1, window_size // 2)

    print(f"Using rolling window of {window_size} days with min_periods={min_periods}")

    # Find key features flexibly
    base_features = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['sleep score', 'resting heart', 'body battery', 'hrv']):
            if df[col].dtype in ['int64', 'float64']:  # Only numeric columns
                base_features.append(col)

    # 7-day rolling averages for found features
    for col in base_features[:4]:  # Limit to first 4 to avoid too many features
        if col in df.columns:
            df[f'{col}_rolling_avg'] = df[col].rolling(window=window_size, min_periods=min_periods).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=min_periods).std()
            print(f"✓ Rolling features created for {col}")

    # Target variable rolling features
    if target_col in df.columns:
        df[f'{target_col}_rolling_avg'] = df[target_col].rolling(window=window_size, min_periods=min_periods).mean()
        df[f'{target_col}_trend'] = df[target_col] - df[f'{target_col}_rolling_avg']
        print(f"✓ Rolling features created for target: {target_col}")

    return df


def create_interaction_features(df):
    """Create interaction features"""
    print("Creating interaction features...")

    # Find relevant columns flexibly
    sleep_score_col = None
    body_battery_col = None
    hrv_col = None
    hr_col = None
    breathing_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'sleep' in col_lower and 'score' in col_lower:
            sleep_score_col = col
        elif 'body' in col_lower and 'battery' in col_lower:
            body_battery_col = col
        elif 'hrv' in col_lower:
            hrv_col = col
        elif 'resting' in col_lower and 'heart' in col_lower:
            hr_col = col
        elif 'breath' in col_lower:
            breathing_col = col

    # Sleep score and body battery interaction
    if sleep_score_col and body_battery_col:
        df['Sleep_Battery_Interaction'] = df[sleep_score_col] * df[body_battery_col] / 100
        print("✓ Sleep-Battery interaction created")

    # HRV and heart rate interaction
    if hrv_col and hr_col:
        df['HRV_HR_Ratio'] = df[hrv_col] / df[hr_col]
        print("✓ HRV-HR ratio created")

    # Breathing and sleep quality
    if breathing_col and sleep_score_col:
        df['Breathing_Sleep_Quality'] = df[sleep_score_col] / df[breathing_col]
        print("✓ Breathing-Sleep quality interaction created")

    return df


def create_lag_features(df, target_col='Deep sleep (mins)'):
    """Create lag features"""
    print("Creating lag features...")

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Find key features flexibly
    key_features = [target_col]  # Always include target

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['sleep score', 'body battery', 'resting heart']):
            if df[col].dtype in ['int64', 'float64'] and col != target_col:
                key_features.append(col)

    # Previous day features
    for feature in key_features[:4]:  # Limit to avoid too many features
        if feature in df.columns:
            df[f'{feature}_prev_day'] = df[feature].shift(1)
            df[f'{feature}_change'] = df[feature] - df[f'{feature}_prev_day']
            print(f"✓ Lag features created for {feature}")

    return df


def get_top_correlated_features(df, target_col='Deep sleep (mins)', top_n=15):
    """Get top N features most correlated with target"""
    print(f"Finding top {top_n} features correlated with {target_col}...")

    # Get numeric columns and exclude problematic columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols
                    if not col.startswith('GMM_outlier')
                    and col != target_col
                    and not col.endswith('_Category')]  # Exclude categorical encoded columns

    # Ensure target exists
    if target_col not in df.columns:
        print(f"Warning: Target column {target_col} not found!")
        return [], pd.Series()

    # Calculate correlations with target, handling NaN values
    correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    correlations = correlations.drop(target_col, errors='ignore')  # Remove self-correlation
    correlations = correlations.dropna()  # Remove NaN correlations

    # Get top N features
    actual_top_n = min(top_n, len(correlations))
    top_features = correlations.head(actual_top_n).index.tolist()
    top_features.append(target_col)  # Add target back for the correlation matrix

    print(f"Found {len(correlations)} valid correlations, showing top {actual_top_n}")

    return top_features, correlations


def create_correlation_analysis(df, target_col='Deep sleep (mins)', top_n=15):
    """Create correlation analysis and visualization"""
    print("Creating correlation analysis...")

    # Get top correlated features
    top_features, all_correlations = get_top_correlated_features(df, target_col, top_n)

    if len(top_features) <= 1:  # Only target or no features
        print("Warning: Not enough features for correlation analysis")
        return pd.DataFrame(), pd.Series()

    # Create correlation matrix for top features
    correlation_matrix = df[top_features].corr()

    # Create visualization
    fig_size = max(12, min(20, len(top_features)))
    plt.figure(figsize=(fig_size, fig_size - 2))

    # Calculate subplot layout
    n_plots = 4 if len(all_correlations) >= 5 else 3

    # Main correlation heatmap
    plt.subplot(2, 2, 1)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=0.5, fmt='.2f',
                cbar_kws={"shrink": .8})
    plt.title(f'Top {len(top_features) - 1} Features Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Target correlation bar plot
    if len(all_correlations) > 0:
        plt.subplot(2, 2, 2)
        display_n = min(top_n, len(all_correlations))
        target_correlations = all_correlations.head(display_n)
        target_correlations.plot(kind='barh', color='steelblue')
        plt.title(f'Top {display_n} Features Correlation with {target_col}')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()

    # Feature importance based on correlation
    if len(all_correlations) >= 5:
        plt.subplot(2, 2, 3)
        feature_importance = all_correlations.sort_values(ascending=True).tail(10)
        feature_importance.plot(kind='barh', color='skyblue')
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Absolute Correlation')

    # Distribution of target variable
    plt.subplot(2, 2, 4)
    df[target_col].hist(bins=min(30, len(df) // 3), alpha=0.7, color='green')
    plt.title(f'Distribution of {target_col}')
    plt.xlabel(target_col)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return correlation_matrix, all_correlations


def save_enhanced_dataset(df, output_path='enhanced_dataset.csv'):
    """Save the enhanced dataset with new features"""
    print(f"Saving enhanced dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Enhanced dataset saved with {df.shape[1]} features and {df.shape[0]} records.")


def print_feature_summary(df, original_cols):
    """Print summary of new features created"""
    new_features = [col for col in df.columns if col not in original_cols]

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 50)
    print(f"Original features: {len(original_cols)}")
    print(f"New features created: {len(new_features)}")
    print(f"Total features: {df.shape[1]}")

    print("\nNew features created:")
    for i, feature in enumerate(new_features, 1):
        print(f"{i:2d}. {feature}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def main():
    """Main execution function"""
    print("Starting Feature Engineering Process...")
    print("=" * 50)

    # Load data
    df = load_and_prepare_data()
    original_columns = df.columns.tolist()
    target_col = 'Deep sleep (mins)'

    # Check if target column exists
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found!")
        print("Available columns:", df.columns.tolist())
        # Try to find a similar column
        sleep_cols = [col for col in df.columns if 'sleep' in col.lower() and 'deep' in col.lower()]
        if sleep_cols:
            target_col = sleep_cols[0]
            print(f"Using '{target_col}' as target column instead")
        else:
            print("No suitable target column found. Please check your data.")
            return None, None, None

    # Basic info about target variable
    print(f"\n🎯 Target Variable: {target_col}")
    print(f"Mean: {df[target_col].mean():.1f}")
    print(f"Std: {df[target_col].std():.1f}")
    print(f"Range: {df[target_col].min():.0f} - {df[target_col].max():.0f}")

    # Perform feature engineering
    if 'Date' in df.columns:
        df = create_time_features(df)
    else:
        print("Skipping time features - no Date column found")

    df = create_sleep_features(df)
    df = create_weather_features(df)

    if 'Date' in df.columns:
        df = create_rolling_features(df, target_col)
        df = create_lag_features(df, target_col)
    else:
        print("Skipping rolling and lag features - no Date column found")

    df = create_interaction_features(df)

    # Print feature summary
    print_feature_summary(df, original_columns)

    # Create correlation analysis
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)

    try:
        correlation_matrix, target_correlations = create_correlation_analysis(df, target_col, top_n=15)

        # Print top correlations if available
        if len(target_correlations) > 0:
            print(f"\nTop 10 features most correlated with {target_col}:")
            for i, (feature, corr) in enumerate(target_correlations.head(10).items(), 1):
                print(f"{i:2d}. {feature:<30} : {corr:.3f}")
        else:
            print("No correlations could be calculated")

    except Exception as e:
        print(f"Error in correlation analysis: {str(e)}")
        correlation_matrix, target_correlations = pd.DataFrame(), pd.Series()

    # Save enhanced dataset
    save_enhanced_dataset(df)

    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Files created:")
    print("- enhanced_dataset.csv (enhanced dataset)")
    print("- correlation_analysis.png (correlation visualization)")

    return df, correlation_matrix, target_correlations


if __name__ == "__main__":
    df, correlation_matrix, target_correlations = main()