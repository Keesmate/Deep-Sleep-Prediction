import pandas as pd

# Load the dataset
df = pd.read_csv('Data_1.csv', parse_dates=['Date'])

# Reformat the Date column
df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

# Save the updated CSV
output_path = 'Data_1.csv'
df.to_csv(output_path, index=False)
