import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
data = pd.read_csv('cleaned_earthquake_catalogue.csv')

# Athens center coordinates
athens_center_lat = 37.9838
athens_center_lon = 23.7275
data['lat_diff'] = data['Latitude'] - athens_center_lat
data['lon_diff'] = data['Longitude'] - athens_center_lon
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

# Features with timestamp
features_with_timestamp = ['Timestamp', 'Latitude', 'Longitude', 'Depth']
# Features without timestamp
features_without_timestamp = ['Latitude', 'Longitude', 'Depth']

# Prepare data
x_with_timestamp = data[features_with_timestamp]
x_without_timestamp = data[features_without_timestamp]
y = data['Magnitude']

# Split data
split_idx = int(len(data) * 0.8)
x_train_with_timestamp = x_with_timestamp.iloc[:split_idx]
x_test_with_timestamp = x_with_timestamp.iloc[split_idx:]
x_train_without_timestamp = x_without_timestamp.iloc[:split_idx]
x_test_without_timestamp = x_without_timestamp.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Scale features
scaler_with_timestamp = StandardScaler()
scaler_without_timestamp = StandardScaler()

x_train_with_timestamp_scaled = scaler_with_timestamp.fit_transform(x_train_with_timestamp)
x_test_with_timestamp_scaled = scaler_with_timestamp.transform(x_test_with_timestamp)
x_train_without_timestamp_scaled = scaler_without_timestamp.fit_transform(x_train_without_timestamp)
x_test_without_timestamp_scaled = scaler_without_timestamp.transform(x_test_without_timestamp)

# Train models
rf_with_timestamp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_without_timestamp = RandomForestRegressor(n_estimators=100, random_state=42)

rf_with_timestamp.fit(x_train_with_timestamp_scaled, y_train)
rf_without_timestamp.fit(x_train_without_timestamp_scaled, y_train)

# Make predictions
y_pred_with_timestamp = rf_with_timestamp.predict(x_test_with_timestamp_scaled)
y_pred_without_timestamp = rf_without_timestamp.predict(x_test_without_timestamp_scaled)

# Calculate R² scores
r2_with_timestamp = r2_score(y_test, y_pred_with_timestamp)
r2_without_timestamp = r2_score(y_test, y_pred_without_timestamp)

# Print results
print(f"R² Score with Timestamp: {r2_with_timestamp:.4f}")
print(f"R² Score without Timestamp: {r2_without_timestamp:.4f}")
print(f"Difference in R² Score: {r2_with_timestamp - r2_without_timestamp:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['With Timestamp', 'Without Timestamp'], 
        [r2_with_timestamp, r2_without_timestamp])
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, v in enumerate([r2_with_timestamp, r2_without_timestamp]):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show() 