import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# Load and prepare data
def load_and_prepare_data(filepath='Data_1.csv'):
    """Load and prepare the dataset with basic preprocessing"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)

    # Print data range info
    print(f"Data range: {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}")
    print(f"Total days: {(df['Date'].max() - df['Date'].min()).days + 1}")
    print(f"Data points: {len(df)}")

    return df


def add_calendar_features(df, date_col='Date'):
    """Add calendar-based features suitable for short-term data (Feb-Jun)"""
    df = df.copy()

    # Basic calendar features
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dow'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['week_of_year'] = df[date_col].dt.isocalendar().week

    # Derived features relevant for short period
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_monday'] = (df['dow'] == 0).astype(int)
    df['is_friday'] = (df['dow'] == 4).astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)

    # Days from start (useful for trend analysis)
    df['days_from_start'] = (df[date_col] - df[date_col].min()).dt.days

    # Week within the dataset
    df['week_number'] = ((df['days_from_start'] // 7) + 1)

    # Day within month (for monthly patterns)
    df['day_of_month'] = df[date_col].dt.day

    return df


def add_fourier_features(df, periods=[7, 14], n_terms=2):
    """Add Fourier features for weekly and bi-weekly patterns"""
    df = df.copy()

    if 'days_from_start' not in df.columns:
        raise ValueError("Need 'days_from_start' column. Run add_calendar_features first.")

    for period in periods:
        period_name = f"p{int(period)}"
        for i in range(1, n_terms + 1):
            df[f'sin_{period_name}_{i}'] = np.sin(2 * np.pi * i * df['days_from_start'] / period)
            df[f'cos_{period_name}_{i}'] = np.cos(2 * np.pi * i * df['days_from_start'] / period)

    return df


def find_dominant_periods(series, max_lag=50, min_correlation=0.1):
    """Find dominant seasonal periods using autocorrelation (limited to short-term patterns)"""
    series_clean = series.dropna()

    # Limit max_lag for short datasets
    max_lag = min(max_lag, len(series_clean) // 3)

    if len(series_clean) < 20:  # Need minimum data points
        return []

    try:
        autocorr = acf(series_clean, nlags=max_lag, fft=True)
        # Look for peaks, focusing on weekly and bi-weekly patterns
        peaks, properties = find_peaks(autocorr[1:], height=min_correlation, distance=3)

        # Filter to realistic periods for short-term data (3-30 days)
        realistic_peaks = peaks[(peaks >= 2) & (peaks <= 30)]

        if len(realistic_peaks) == 0:
            return []

        # Return periods sorted by correlation strength
        peak_correlations = autocorr[realistic_peaks + 1]
        sorted_indices = np.argsort(peak_correlations)[::-1]
        dominant_periods = (realistic_peaks[sorted_indices] + 1).tolist()

        return dominant_periods[:3]  # Return top 3 periods
    except:
        return []


def calculate_seasonality_strength(seasonal, residual):
    """Calculate improved seasonality strength metric"""
    try:
        seasonal_var = seasonal.var()
        residual_var = residual.var()

        if seasonal_var + residual_var == 0:
            return 0

        # STL-style strength calculation
        strength = max(0, 1 - residual_var / (seasonal_var + residual_var))
        return strength
    except:
        return 0


def test_seasonality_significance(series, period):
    """Test statistical significance of seasonality using Kruskal-Wallis test"""
    try:
        series_clean = series.dropna()
        if len(series_clean) < 2 * period:
            return False

        # Group data by seasonal cycle
        groups = []
        for i in range(period):
            group = series_clean.iloc[i::period]
            if len(group) > 1:
                groups.append(group)

        if len(groups) < 2:
            return False

        # Kruskal-Wallis test
        statistic, p_value = stats.kruskal(*groups)
        return p_value < 0.05
    except:
        return False


def decompose_series(series, period, method='additive', model_type='seasonal_decompose'):
    """Decompose time series using specified method"""
    try:
        if model_type == 'stl' and len(series) >= 2 * period:
            stl = STL(series, period=period, seasonal=7)
            result = stl.fit()
            return result
        else:
            if len(series) >= 2 * period:
                result = seasonal_decompose(series, period=period, model=method)
                return result
    except Exception as e:
        print(f"Decomposition failed for period {period}: {e}")
        return None


def comprehensive_seasonal_analysis(df, date_col='Date',
                                    periods=[7, 14],  # Focus on weekly and bi-weekly
                                    strength_threshold=0.1,
                                    auto_detect_periods=True):
    """
    Comprehensive seasonal analysis optimized for short-term data (Feb-Jun)
    Only analyzes original data features, excluding engineered calendar/Fourier features
    """
    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col])
    df_work = df_work.sort_values(date_col).set_index(date_col)

    seasonal_results = {}

    # Get only ORIGINAL data columns, exclude engineered features
    all_numeric_cols = df_work.select_dtypes(include=[np.number]).columns

    # Define engineered feature patterns to exclude
    engineered_patterns = [
        'month', 'day', 'dow', 'week_of_year', 'quarter',
        'is_weekend', 'is_monday', 'is_friday', 'is_month_end', 'is_month_start',
        'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start',
        'days_from_start', 'week_number', 'day_of_month',
        'sin_p', 'cos_p'  # Fourier features
    ]

    # Filter to only original data columns
    numeric_cols = []
    for col in all_numeric_cols:
        is_engineered = any(pattern in col for pattern in engineered_patterns)
        if not is_engineered:
            numeric_cols.append(col)

    numeric_cols = [col for col in numeric_cols]  # Convert to list

    print("=" * 60)
    print("SEASONAL ANALYSIS RESULTS")
    print("=" * 60)

    # Summary statistics
    data_length = len(df_work)
    print(f"Dataset length: {data_length} days")
    print(f"Total numeric columns: {len(all_numeric_cols)}")
    print(f"Analyzing original data features: {len(numeric_cols)}")
    print(f"Excluded engineered features: {len(all_numeric_cols) - len(numeric_cols)}")
    print(f"Testing periods: {periods}")
    print(f"\nOriginal data columns being analyzed:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"  {i:2d}. {col}")
    print()

    for col in numeric_cols:
        print(f"\nAnalyzing variable: {col}")
        print("-" * 40)

        series = df_work[col].dropna()
        col_results = {}

        # Auto-detect periods if enabled
        test_periods = periods.copy()
        if auto_detect_periods:
            detected_periods = find_dominant_periods(series)
            if detected_periods:
                print(f"Auto-detected periods: {detected_periods}")
                test_periods.extend([p for p in detected_periods if p not in test_periods])

        # Test each period
        best_strength = 0
        best_config = None

        for period in test_periods:
            if len(series) >= 2 * period and period <= len(series) // 3:

                # Test both additive and multiplicative models
                for model in ['additive', 'multiplicative']:
                    try:
                        decomp = decompose_series(series, period=period, method=model)

                        if decomp is not None:
                            # Calculate strength
                            strength = calculate_seasonality_strength(decomp.seasonal, decomp.resid)

                            # Test statistical significance
                            is_significant = test_seasonality_significance(series, period)

                            config_key = f'{model}_{period}'
                            col_results[config_key] = {
                                'strength': strength,
                                'significant': is_significant,
                                'decomposition': decomp,
                                'period': period,
                                'model': model
                            }

                            print(f"  {model.capitalize()} model, period {period}:")
                            print(f"    Strength: {strength:.3f}")
                            print(f"    Significant: {is_significant}")

                            # Track best configuration
                            if strength > best_strength and is_significant:
                                best_strength = strength
                                best_config = config_key

                            # Add features if strong and significant
                            if strength >= strength_threshold and is_significant:
                                df_work[f'{col}_seasonal_{model}_p{period}'] = decomp.seasonal
                                df_work[f'{col}_trend_{model}_p{period}'] = decomp.trend
                                print(f"    -> Added seasonal and trend features")

                    except Exception as e:
                        print(f"    Error with {model} model, period {period}: {e}")

        # Report best configuration
        if best_config:
            print(f"\n  Best configuration: {best_config} (strength: {best_strength:.3f})")
        else:
            print(f"\n  No significant seasonality detected for {col}")

        seasonal_results[col] = col_results

    return df_work.reset_index(), seasonal_results


def clean_dataset_for_modeling(df, seasonal_results):
    """
    Clean the dataset by removing unnecessary columns and keeping only modeling-ready features
    """
    df_clean = df.copy()

    print("\n🧹 CLEANING DATASET FOR MODELING")
    print("=" * 50)
    original_cols = len(df_clean.columns)

    # 1. Remove duplicate column names
    print("   • Checking for duplicate column names...")
    if df_clean.columns.duplicated().any():
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        print(f"   • Removed duplicate column names")

    # 2. Remove zero-variance and constant features
    print("   • Checking for zero/constant variance features...")
    cols_to_remove = []

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            if df_clean[col].nunique() <= 1:  # Constant values
                cols_to_remove.append(col)
            elif df_clean[col].var() == 0:  # Zero variance
                cols_to_remove.append(col)
        except:
            continue

    if cols_to_remove:
        df_clean = df_clean.drop(columns=cols_to_remove)
        print(f"   • Removed {len(cols_to_remove)} zero/constant variance columns")

    # 3. Handle duplicate seasonal features - keep only the best model
    print("   • Handling duplicate seasonal features...")
    seasonal_cols = [col for col in df_clean.columns if '_seasonal_' in col or '_trend_' in col]

    # Group by base variable and period
    seasonal_groups = {}
    for col in seasonal_cols:
        try:
            if '_seasonal_' in col:
                base_var = col.split('_seasonal_')[0]
                rest = col.split('_seasonal_')[1]
            elif '_trend_' in col:
                base_var = col.split('_trend_')[0]
                rest = col.split('_trend_')[1]
            else:
                continue

            # Extract period (e.g., "additive_p29" -> "29")
            if '_p' in rest:
                period = rest.split('_p')[1]
                key = f"{base_var}_p{period}"

                if key not in seasonal_groups:
                    seasonal_groups[key] = []
                seasonal_groups[key].append(col)
        except:
            continue

    # Remove inferior seasonal features
    seasonal_to_remove = []
    for key, feature_list in seasonal_groups.items():
        if len(feature_list) > 2:  # Multiple models for same variable/period
            base_var = key.split('_p')[0]

            # Find best model from seasonal_results
            if base_var in seasonal_results and seasonal_results[base_var]:
                try:
                    best_result = max(seasonal_results[base_var].items(),
                                      key=lambda x: x[1]['strength'])
                    best_model = best_result[1]['model']

                    # Remove features from inferior models
                    for col in feature_list:
                        if best_model not in col:
                            seasonal_to_remove.append(col)
                except:
                    continue

    if seasonal_to_remove:
        df_clean = df_clean.drop(columns=seasonal_to_remove)
        print(f"   • Removed {len(seasonal_to_remove)} inferior seasonal features")

    # 4. Remove weak seasonal features
    print("   • Removing weak seasonal features...")
    remaining_seasonal = [col for col in df_clean.columns if '_seasonal_' in col or '_trend_' in col]
    weak_seasonal = []

    for col in remaining_seasonal:
        try:
            if '_seasonal_' in col:
                base_var = col.split('_seasonal_')[0]
            elif '_trend_' in col:
                base_var = col.split('_trend_')[0]
            else:
                continue

            if base_var in seasonal_results and seasonal_results[base_var]:
                best_result = max(seasonal_results[base_var].items(),
                                  key=lambda x: x[1]['strength'])
                if best_result[1]['strength'] < 0.1 or not best_result[1]['significant']:
                    weak_seasonal.append(col)
        except:
            continue

    if weak_seasonal:
        df_clean = df_clean.drop(columns=weak_seasonal)
        print(f"   • Removed {len(weak_seasonal)} weak seasonal features")

    # 5. Handle Fourier features based on detected seasonality
    print("   • Managing Fourier features...")
    fourier_cols = [col for col in df_clean.columns if col.startswith(('sin_p', 'cos_p'))]

    # Find periods with strong seasonality
    strong_periods = set()
    for var, results in seasonal_results.items():
        if results:
            try:
                best = max(results.items(), key=lambda x: x[1]['strength'])
                if best[1]['strength'] >= 0.1 and best[1]['significant']:
                    strong_periods.add(best[1]['period'])
            except:
                continue

    # Remove Fourier features for periods without strong seasonality
    fourier_to_remove = []
    if not strong_periods:
        fourier_to_remove = fourier_cols
        print(f"   • Removing all {len(fourier_cols)} Fourier features (no strong seasonality)")
    else:
        for col in fourier_cols:
            try:
                # Extract period from column name (e.g., sin_p7_1 -> 7)
                period = int(col.split('_p')[1].split('_')[0])
                if period not in strong_periods:
                    fourier_to_remove.append(col)
            except:
                fourier_to_remove.append(col)

        if fourier_to_remove:
            print(f"   • Removing {len(fourier_to_remove)} Fourier features for weak periods")
        if strong_periods:
            print(f"   • Keeping Fourier features for periods: {sorted(strong_periods)}")

    if fourier_to_remove:
        df_clean = df_clean.drop(columns=fourier_to_remove)

    # 6. Remove redundant calendar features
    print("   • Removing redundant calendar features...")
    redundant_features = []

    # Remove features not relevant for 3.5-month dataset
    redundant_patterns = ['quarter', 'is_quarter', 'is_year', 'week_of_year']
    for col in df_clean.columns:
        if any(pattern in col for pattern in redundant_patterns):
            redundant_features.append(col)

    # Handle day vs day_of_month duplication
    if 'day' in df_clean.columns and 'day_of_month' in df_clean.columns:
        try:
            if df_clean['day'].equals(df_clean['day_of_month']):
                redundant_features.append('day')
                print(f"   • 'day' and 'day_of_month' are identical - removing 'day'")
        except:
            pass

    if redundant_features:
        existing_redundant = [col for col in redundant_features if col in df_clean.columns]
        if existing_redundant:
            df_clean = df_clean.drop(columns=existing_redundant)
            print(f"   • Removed {len(existing_redundant)} redundant features")

    # 7. Final cleanup - remove any NaN-only columns
    print("   • Final cleanup...")
    nan_cols = []
    for col in df_clean.columns:
        if col != 'Date' and df_clean[col].isna().all():
            nan_cols.append(col)

    if nan_cols:
        df_clean = df_clean.drop(columns=nan_cols)
        print(f"   • Removed {len(nan_cols)} all-NaN columns")

    # 8. Organize columns logically
    print("   • Organizing columns...")

    # Categorize remaining columns
    date_cols = ['Date']
    original_data_cols = []
    time_encoding_cols = []
    calendar_cols = []
    fourier_cols = []
    seasonal_cols = []
    outlier_cols = []

    for col in df_clean.columns:
        if col == 'Date':
            continue
        elif col.startswith('GMM_outlier_'):
            outlier_cols.append(col)
        elif col.endswith(('Sin', 'Cos')) and not col.startswith(('sin_p', 'cos_p')):
            time_encoding_cols.append(col)
        elif col.startswith(('sin_p', 'cos_p')):
            fourier_cols.append(col)
        elif '_seasonal_' in col or '_trend_' in col:
            seasonal_cols.append(col)
        elif any(pattern in col for pattern in ['month', 'dow', 'week_number', 'day_of_month',
                                                'is_weekend', 'is_monday', 'is_friday',
                                                'is_month_end', 'is_month_start', 'days_from_start']):
            calendar_cols.append(col)
        else:
            original_data_cols.append(col)

    # Reorder columns
    column_order = (date_cols + original_data_cols + time_encoding_cols +
                    calendar_cols + fourier_cols + seasonal_cols + outlier_cols)

    # Make sure all columns exist
    final_columns = [col for col in column_order if col in df_clean.columns]
    df_clean = df_clean[final_columns]

    # 9. Summary report
    total_removed = original_cols - len(df_clean.columns)
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"   • Original columns: {original_cols}")
    print(f"   • Final columns: {len(df_clean.columns)}")
    print(f"   • Removed columns: {total_removed}")
    print(f"   • Reduction: {total_removed / original_cols * 100:.1f}%")

    print(f"\n📋 FINAL DATASET COMPOSITION:")
    print(f"   📈 Original Data: {len(original_data_cols)} features")
    print(f"   🕐 Time Encoding: {len(time_encoding_cols)} features")
    print(f"   📅 Calendar: {len(calendar_cols)} features")
    print(f"   🌊 Fourier: {len(fourier_cols)} features")
    print(f"   📈 Seasonal: {len(seasonal_cols)} features")
    print(f"   🚨 Outlier Detection: {len(outlier_cols)} features")

    return df_clean


def plot_seasonal_analysis(df, seasonal_results, max_plots=5):
    """Plot seasonal decomposition for variables with strongest seasonality"""

    # Find variables with strongest seasonality
    var_strengths = []
    for col, results in seasonal_results.items():
        if results:
            best_result = max(results.items(), key=lambda x: x[1]['strength'])
            if best_result[1]['significant']:
                var_strengths.append((col, best_result[0], best_result[1]['strength']))

    # Sort by strength and plot top variables
    var_strengths.sort(key=lambda x: x[2], reverse=True)

    for i, (col, config, strength) in enumerate(var_strengths[:max_plots]):
        decomp = seasonal_results[col][config]['decomposition']

        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        decomp.observed.plot(ax=axes[0], title=f'{col} - Original Series')
        decomp.trend.plot(ax=axes[1], title='Trend Component', color='orange')
        decomp.seasonal.plot(ax=axes[2], title='Seasonal Component', color='green')
        decomp.resid.plot(ax=axes[3], title='Residual Component', color='red')

        plt.suptitle(f'Seasonal Decomposition: {col}\n{config} (Strength: {strength:.3f})',
                     fontsize=14, y=0.98)
        plt.tight_layout()
        plt.show()


def plot_calendar_patterns(df, numeric_cols=None, max_cols=6):
    """Plot patterns by day of week and week number for original data features only"""
    if numeric_cols is None:
        # Get only original data columns, exclude engineered features
        all_numeric_cols = df.select_dtypes(include=[np.number]).columns

        engineered_patterns = [
            'month', 'day', 'dow', 'week_of_year', 'quarter',
            'is_weekend', 'is_monday', 'is_friday', 'is_month_end', 'is_month_start',
            'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start',
            'days_from_start', 'week_number', 'day_of_month',
            'sin_p', 'cos_p', '_seasonal_', '_trend_'  # Include decomposition features
        ]

        numeric_cols = []
        for col in all_numeric_cols:
            is_engineered = any(pattern in col for pattern in engineered_patterns)
            if not is_engineered:
                numeric_cols.append(col)

    numeric_cols = numeric_cols[:max_cols]  # Limit plots

    print(f"\nPlotting calendar patterns for {len(numeric_cols)} original data features:")
    for col in numeric_cols:
        print(f"  - {col}")

    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Day of week pattern
        if 'dow' in df.columns:
            sns.boxplot(data=df, x='dow', y=col, ax=axes[0])
            axes[0].set_title(f'{col} by Day of Week')
            axes[0].set_xlabel('Day of Week (0=Mon, 6=Sun)')

        # Week number pattern
        if 'week_number' in df.columns:
            sns.boxplot(data=df, x='week_number', y=col, ax=axes[1])
            axes[1].set_title(f'{col} by Week Number')
            axes[1].set_xlabel('Week Number in Dataset')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


def generate_seasonal_summary(seasonal_results, threshold=0.1):
    """Generate a summary of seasonal analysis results for original data features"""
    summary_data = []

    for col, results in seasonal_results.items():
        if results:
            best_result = max(results.items(), key=lambda x: x[1]['strength'])
            config, details = best_result

            summary_data.append({
                'Variable': col,
                'Best_Config': config,
                'Strength': details['strength'],
                'Significant': details['significant'],
                'Period': details['period'],
                'Model': details['model'],
                'Strong_Seasonal': details['strength'] >= threshold and details['significant']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Strength', ascending=False)

    print("\n" + "=" * 80)
    print("SEASONAL ANALYSIS SUMMARY - ORIGINAL DATA FEATURES ONLY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    strong_seasonal = summary_df[summary_df['Strong_Seasonal']]['Variable'].tolist()
    significant_any = summary_df[summary_df['Significant']]['Variable'].tolist()

    print(f"\n📊 SUMMARY OF FINDINGS:")
    print(f"   • Variables analyzed: {len(summary_df)}")
    print(f"   • Variables with significant seasonality: {len(significant_any)}")
    print(f"   • Variables with strong seasonality (≥{threshold}): {len(strong_seasonal)}")

    if strong_seasonal:
        print(f"\n🔍 STRONG SEASONAL PATTERNS DETECTED:")
        for var in strong_seasonal:
            row = summary_df[summary_df['Variable'] == var].iloc[0]
            print(f"   • {var}: {row['Period']}-day cycle (strength: {row['Strength']:.3f})")
    else:
        print(f"\n⚠️  No strong seasonal patterns detected in original data features.")
        if significant_any:
            print(f"   However, {len(significant_any)} variables show statistically significant (but weak) patterns:")
            for var in significant_any:
                row = summary_df[summary_df['Variable'] == var].iloc[0]
                print(f"   • {var}: {row['Period']}-day cycle (strength: {row['Strength']:.3f})")

    return summary_df


# Main execution function
def main():
    """Main function to run the complete seasonal analysis on original data features only"""

    print("🔍 SEASONAL PATTERN ANALYSIS")
    print("=" * 50)
    print("This analysis focuses on discovering genuine seasonal patterns")
    print("in your original data features (health metrics, weather, etc.)")
    print("Engineered calendar features are excluded from analysis.")
    print()

    # Load data
    print("📂 Loading data...")
    df = load_and_prepare_data('Data_1.csv')

    # Add calendar features (for use in analysis, not for seasonal detection)
    print("📅 Adding calendar features...")
    df = add_calendar_features(df)

    # Add Fourier features for weekly patterns (for use in modeling, not for seasonal detection)
    print("🌊 Adding Fourier features...")
    df = add_fourier_features(df, periods=[7, 14], n_terms=2)

    # Run comprehensive seasonal analysis on ORIGINAL data only
    print("🔍 Running seasonal analysis on original data features...")
    df_enhanced, seasonal_results = comprehensive_seasonal_analysis(
        df,
        periods=[7, 14],  # Weekly and bi-weekly for short dataset
        strength_threshold=0.1,
        auto_detect_periods=True
    )

    # Generate summary
    summary_df = generate_seasonal_summary(seasonal_results)

    # Clean dataset for modeling
    df_clean = clean_dataset_for_modeling(df_enhanced, seasonal_results)

    # Plot results
    print("\n📊 Generating plots...")
    plot_calendar_patterns(df_clean)  # Use cleaned dataset for plots
    plot_seasonal_analysis(df_clean, seasonal_results)

    # Save cleaned dataset
    output_file = 'Data_1_seasonal_features_clean.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n💾 Clean modeling dataset saved to: {output_file}")

    # Final recommendations
    strong_vars = summary_df[summary_df['Strong_Seasonal']]['Variable'].tolist()
    if strong_vars:
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   • Consider using the detected seasonal patterns for forecasting")
        print(f"   • Seasonal features have been added to the dataset for these variables:")
        for var in strong_vars:
            row = summary_df[summary_df['Variable'] == var].iloc[0]
            print(f"     - {var}_seasonal_{row['Model']}_p{row['Period']}")
            print(f"     - {var}_trend_{row['Model']}_p{row['Period']}")
        print(f"   • Use calendar features (dow, week_number) for modeling weekly patterns")
        print(f"   • Consider Fourier features for smooth seasonal representations")
    else:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Limited seasonal patterns detected in your 3.5-month dataset")
        print(f"   • This could be due to:")
        print(f"     - Short observation period (need more data for strong patterns)")
        print(f"     - Variables genuinely lacking strong seasonality")
        print(f"     - Patterns longer than your observation window")
        print(f"   • Consider using calendar features for weekly modeling")
        print(f"   • Collect more data over longer periods for better seasonal detection")

    print(f"\n✅ Analysis complete! Clean dataset ready for modeling.")

    return df_clean, seasonal_results, summary_df


# Run the analysis
if __name__ == "__main__":
    df_clean, seasonal_results, summary_df = main()