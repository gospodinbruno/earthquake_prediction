import pandas as pd

# Define correct column names manually
columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Latitude', 'Longitude', 'Depth_km', 'Magnitude']

# Read the file, skipping 3 junk header rows, and manually assign the right column names
df = pd.read_csv('earthquake_catalogue.csv', skiprows=3, names=columns)

# Drop rows with any NaN values
df = df.dropna()

# (Optional) Reset the index after dropping
df = df.reset_index(drop=True)

# Month abbreviation to number mapping
month_mapping = {
    'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
}

# Replace month names with numbers
df['Month'] = df['Month'].map(month_mapping)

# Create the new Date column
df['Date'] = df['Year'].astype(str) + '-' + df['Month'] + '-' + df['Day'].astype(str).str.zfill(2)

# Select only the needed columns
final_df = df[['Date', 'Latitude', 'Longitude', 'Magnitude', 'Depth_km']]

# Rename Depth_km to Depth
final_df = final_df.rename(columns={'Depth_km': 'Depth'})

# Save to CSV
final_df.to_csv('cleaned_earthquake_catalogue.csv', index=False)

print("✅ Cleaned CSV created successfully!")
