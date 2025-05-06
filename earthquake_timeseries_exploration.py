import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

# Load data
data = pd.read_csv('cleaned_earthquake_catalogue.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data = data.sort_values('Datetime').reset_index(drop=True)

# Optional: Filter out magnitudes < 2.0
data = data[data['Magnitude'] >= 2.0]

# -----------------------------------------------
# SECTION 1: Time Series Feature Engineering
# -----------------------------------------------

# Lagged features (autoregressive)
data['Magnitude_t-1'] = data['Magnitude'].shift(1)
data['Magnitude_t-2'] = data['Magnitude'].shift(2)

# Rolling features (mean magnitude in past 3 quakes)
data['RollingMean3'] = data['Magnitude'].rolling(window=3).mean()

# Drop NA rows (first few due to shift/rolling)
data = data.dropna().reset_index(drop=True)

# -----------------------------------------------
# SECTION 2: Seasonal Patterns & Time Breakdown
# -----------------------------------------------

data['Year'] = data['Datetime'].dt.year
data['Month'] = data['Datetime'].dt.month
data['DayOfWeek'] = data['Datetime'].dt.dayofweek
data['Hour'] = data['Datetime'].dt.hour

# Earthquake count by year
plt.figure(figsize=(10, 5))
data.groupby('Year').size().plot(kind='bar')
plt.title('Earthquake Count by Year')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig('yearly_quake_counts.png')
plt.show()

# Average magnitude by month
plt.figure(figsize=(10, 5))
data.groupby('Month')['Magnitude'].mean().plot(kind='bar', color='orange')
plt.title('Average Earthquake Magnitude by Month')
plt.ylabel('Average Magnitude')
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_avg_magnitude.png')
plt.show()

# -----------------------------------------------
# SECTION 3: Autocorrelation Analysis
# -----------------------------------------------

plt.figure(figsize=(10, 5))
autocorrelation_plot(data['Magnitude'])
plt.title('Autocorrelation of Earthquake Magnitudes')
plt.tight_layout()
plt.savefig('magnitude_autocorrelation.png')
plt.show()

# -----------------------------------------------
# SECTION 4: LSTM-Ready Data Preview
# -----------------------------------------------

# Select time series columns
lstm_data = data[['Datetime', 'Magnitude_t-2', 'Magnitude_t-1', 'Magnitude']].copy()

# Normalize just for LSTM (optional)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
lstm_scaled = scaler.fit_transform(lstm_data[['Magnitude_t-2', 'Magnitude_t-1', 'Magnitude']])
lstm_scaled = pd.DataFrame(lstm_scaled, columns=['t-2', 't-1', 't'])

print("\n🔢 LSTM-Ready Sample Data:")
print(lstm_scaled.head())

# Save to file (optional)
lstm_scaled.to_csv("lstm_ready_dataset.csv", index=False)
