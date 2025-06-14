import pandas as pd
import numpy as np


def inspect_csv_dates(filepath='Data_1.csv'):
    """Inspect the CSV file to understand the date format and identify issues"""

    print("🔍 INSPECTING CSV FILE FOR DATE FORMAT ISSUES")
    print("=" * 60)

    # First, let's read the raw data without any date parsing
    try:
        # Read first few rows to understand structure
        df_raw = pd.read_csv(filepath, nrows=10)
        print("✅ CSV file loaded successfully")
        print(f"📊 Dataset shape (first 10 rows): {df_raw.shape}")
        print(f"📝 Column names: {list(df_raw.columns)}")

        # Check if 'Date' column exists
        if 'Date' in df_raw.columns:
            print(f"\n📅 DATE COLUMN ANALYSIS:")
            print(f"   • Date column found: ✅")
            print(f"   • Sample date values:")

            # Show first 10 date values
            for i, date_val in enumerate(df_raw['Date'].head(10)):
                print(f"     Row {i + 1}: '{date_val}' (type: {type(date_val)})")

            # Check for missing values
            date_series = df_raw['Date']
            if date_series.isna().any():
                print(f"   • Missing values: {date_series.isna().sum()} found ⚠️")
            else:
                print(f"   • Missing values: None ✅")

            # Analyze date patterns
            print(f"\n🔍 DATE PATTERN ANALYSIS:")
            unique_lengths = df_raw['Date'].astype(str).str.len().value_counts()
            print(f"   • Character lengths in date column:")
            for length, count in unique_lengths.items():
                print(f"     - Length {length}: {count} occurrences")

            # Try to identify date format
            sample_dates = df_raw['Date'].dropna().astype(str).head(5).tolist()
            print(f"   • Sample dates for format identification:")
            for i, date_str in enumerate(sample_dates):
                print(f"     {i + 1}. '{date_str}'")

        else:
            print(f"❌ No 'Date' column found!")
            print(f"Available columns: {list(df_raw.columns)}")
            return None

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None

    return df_raw


def try_different_date_formats(df_raw):
    """Try parsing dates with different formats"""

    if 'Date' not in df_raw.columns:
        print("No Date column to parse")
        return None

    print(f"\n🧪 TESTING DIFFERENT DATE PARSING METHODS")
    print("=" * 60)

    date_column = df_raw['Date'].copy()

    # Method 1: Let pandas infer the format
    try:
        print("Method 1: Auto-inference...")
        dates_auto = pd.to_datetime(date_column, errors='coerce')
        success_rate = (1 - dates_auto.isna().sum() / len(dates_auto)) * 100
        print(f"   • Success rate: {success_rate:.1f}%")
        if success_rate > 90:
            print(f"   • ✅ AUTO-INFERENCE WORKS WELL")
            print(f"   • Sample parsed dates: {dates_auto.dropna().head(3).tolist()}")
            return 'auto'
        else:
            print(f"   • ⚠️ Many parsing failures")
    except Exception as e:
        print(f"   • ❌ Failed: {e}")

    # Method 2: dayfirst=True
    try:
        print("\nMethod 2: dayfirst=True...")
        dates_dayfirst = pd.to_datetime(date_column, dayfirst=True, errors='coerce')
        success_rate = (1 - dates_dayfirst.isna().sum() / len(dates_dayfirst)) * 100
        print(f"   • Success rate: {success_rate:.1f}%")
        if success_rate > 90:
            print(f"   • ✅ DAYFIRST=TRUE WORKS WELL")
            print(f"   • Sample parsed dates: {dates_dayfirst.dropna().head(3).tolist()}")
            return 'dayfirst'
        else:
            print(f"   • ⚠️ Many parsing failures")
    except Exception as e:
        print(f"   • ❌ Failed: {e}")

    # Method 3: format='mixed'
    try:
        print("\nMethod 3: format='mixed'...")
        dates_mixed = pd.to_datetime(date_column, format='mixed', errors='coerce')
        success_rate = (1 - dates_mixed.isna().sum() / len(dates_mixed)) * 100
        print(f"   • Success rate: {success_rate:.1f}%")
        if success_rate > 90:
            print(f"   • ✅ MIXED FORMAT WORKS WELL")
            print(f"   • Sample parsed dates: {dates_mixed.dropna().head(3).tolist()}")
            return 'mixed'
        else:
            print(f"   • ⚠️ Many parsing failures")
    except Exception as e:
        print(f"   • ❌ Failed: {e}")

    # Method 4: Common date formats
    common_formats = [
        '%Y-%m-%d',  # 2024-02-15
        '%d/%m/%Y',  # 15/02/2024
        '%m/%d/%Y',  # 02/15/2024
        '%Y/%m/%d',  # 2024/02/15
        '%d-%m-%Y',  # 15-02-2024
        '%m-%d-%Y',  # 02-15-2024
        '%d.%m.%Y',  # 15.02.2024
        '%m.%d.%Y',  # 02.15.2024
    ]

    print(f"\nMethod 4: Testing common formats...")
    for fmt in common_formats:
        try:
            dates_fmt = pd.to_datetime(date_column, format=fmt, errors='coerce')
            success_rate = (1 - dates_fmt.isna().sum() / len(dates_fmt)) * 100
            print(f"   • Format '{fmt}': {success_rate:.1f}% success")
            if success_rate > 90:
                print(f"   • ✅ FORMAT '{fmt}' WORKS WELL")
                print(f"   • Sample parsed dates: {dates_fmt.dropna().head(3).tolist()}")
                return fmt
        except Exception as e:
            print(f"   • Format '{fmt}': Failed - {e}")

    print(f"\n❌ No reliable date parsing method found")
    return None


def create_fixed_load_function(best_method):
    """Create a corrected load_and_prepare_data function"""

    if best_method == 'auto':
        date_parse_code = "df['Date'] = pd.to_datetime(df['Date'])"
    elif best_method == 'dayfirst':
        date_parse_code = "df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)"
    elif best_method == 'mixed':
        date_parse_code = "df['Date'] = pd.to_datetime(df['Date'], format='mixed')"
    elif isinstance(best_method, str) and '%' in best_method:
        date_parse_code = f"df['Date'] = pd.to_datetime(df['Date'], format='{best_method}')"
    else:
        date_parse_code = "df['Date'] = pd.to_datetime(df['Date'], errors='coerce')"

    fixed_function = f'''
def load_and_prepare_data(filepath='Data_1.csv'):
    """Load and prepare the dataset with basic preprocessing - FIXED VERSION"""
    df = pd.read_csv(filepath)

    # FIXED DATE PARSING:
    {date_parse_code}

    # Remove any rows where date parsing failed
    initial_rows = len(df)
    df = df.dropna(subset=['Date'])
    if len(df) < initial_rows:
        print(f"⚠️ Removed {{initial_rows - len(df)}} rows with unparseable dates")

    df = df.sort_values('Date').reset_index(drop=True)

    # Print data range info
    print(f"Data range: {{df['Date'].min().strftime('%d/%m/%Y')}} to {{df['Date'].max().strftime('%d/%m/%Y')}}")
    print(f"Total days: {{(df['Date'].max() - df['Date'].min()).days + 1}}")
    print(f"Data points: {{len(df)}}")

    return df
'''

    return fixed_function


# Main debugging function
def debug_date_parsing(filepath='Data_1.csv'):
    """Main function to debug and fix date parsing issues"""

    # Step 1: Inspect the CSV
    df_raw = inspect_csv_dates(filepath)
    if df_raw is None:
        return None

    # Step 2: Try different parsing methods
    best_method = try_different_date_formats(df_raw)

    if best_method:
        print(f"\n🎯 RECOMMENDED SOLUTION")
        print("=" * 40)
        print(f"✅ Best parsing method: {best_method}")

        # Step 3: Create fixed function
        fixed_function = create_fixed_load_function(best_method)
        print(f"\n📝 CORRECTED LOAD FUNCTION:")
        print(fixed_function)

        return best_method
    else:
        print(f"\n🚨 MANUAL INSPECTION NEEDED")
        print("=" * 40)
        print("Please manually check your Date column for:")
        print("• Inconsistent date formats")
        print("• Non-date values mixed in")
        print("• Special characters or extra spaces")
        print("• Different separators (/, -, .)")

        return None


# Run the debugging
if __name__ == "__main__":
    result = debug_date_parsing('Data_1.csv')