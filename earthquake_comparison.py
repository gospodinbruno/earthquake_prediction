import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor  # Updated API

# --- Load and Prepare Data ---
data = pd.read_csv('cleaned_earthquake_catalogue.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

features = ['Timestamp', 'Latitude', 'Longitude', 'Depth']
x = data[features]
y = data['Magnitude']

# Take a random sample of 50,000 records
data_sample = data.sample(n=50000, random_state=42)

# Sort the sampled data chronologically
data_sample = data_sample.sort_values(by='Timestamp').reset_index(drop=True)

# Chronological split: 80% train, 20% test
split_idx = int(len(data_sample) * 0.8)  # 40,000 train, 10,000 test

x_train = data_sample[features].iloc[:split_idx]
y_train = data_sample['Magnitude'].iloc[:split_idx]

x_test = data_sample[features].iloc[split_idx:]
y_test = data_sample['Magnitude'].iloc[split_idx:]
# --- Correct Scaling (Only Fit on Train) ---

print("First 5 Timestamps (Train Start):")
print(data_sample['Datetime'].head())

print("\nLast 5 Timestamps (Test End):")
print(data_sample['Datetime'].tail())


# Magnitude Range Counts
magnitude_ranges = {
    'Minor (< 2.0)': (0, 2.0),
    'Light (2.0 - 3.0)': (2.0, 3.0),
    'Moderate (3.0 - 4.0)': (3.0, 4.0),
    'Strong (≥ 4.0)': (4.0, float('inf'))
}

print("\n📊 Magnitude Range Distribution (in Sampled Data):")
for range_name, (min_mag, max_mag) in magnitude_ranges.items():
    count = ((data_sample['Magnitude'] >= min_mag) & (data_sample['Magnitude'] < max_mag)).sum()
    print(f"{range_name}: {count} samples")

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- Neural Network Definition ---
def create_nn_model():
    model = Sequential([
        Dense(64, input_dim=x_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# --- Models ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=3),
    'Neural Network': KerasRegressor(model=create_nn_model, epochs=50, batch_size=32, verbose=0)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2, 'Predictions': y_pred}
    print(f"Finished {name}")

# --- Visualize Metrics ---
metrics_df = pd.DataFrame({
    name: {'MSE': results[name]['MSE'], 'MAE': results[name]['MAE'], 'R2': results[name]['R2']}
    for name in models.keys()
}).T

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.barplot(x=metrics_df.index, y=metrics_df['MSE'], palette='viridis')
plt.title('Mean Squared Error by Model')
plt.xticks(rotation=45)
plt.ylabel('MSE')

plt.subplot(2, 2, 2)
sns.barplot(x=metrics_df.index, y=metrics_df['MAE'], palette='viridis')
plt.title('Mean Absolute Error by Model')
plt.xticks(rotation=45)
plt.ylabel('MAE')

plt.subplot(2, 2, 3)
sns.barplot(x=metrics_df.index, y=metrics_df['R2'], palette='viridis')
plt.title('R² Score by Model')
plt.xticks(rotation=45)
plt.ylabel('R²')

plt.tight_layout()
plt.show()



# ===============================
# Additional Error Analysis Plots
# ===============================

# Calculate intuitive metrics
for name, model in models.items():
    y_pred = results[name]['Predictions']
    within_point_five = np.mean(abs(y_test - y_pred) <= 0.5) * 100
    avg_diff = np.mean(abs(y_test - y_pred))
    
    results[name].update({
        'Within_0.5': within_point_five,
        'Avg_Diff': avg_diff,
    })

intuitive_metrics = pd.DataFrame({
    name: {
        'Predictions within 0.5 magnitude (%)': results[name]['Within_0.5'],
        'Average prediction error': results[name]['Avg_Diff'],
    } for name in models.keys()
}).T

# Plot Predictions within 0.5 magnitude
plt.figure(figsize=(10, 6))
sns.barplot(x=intuitive_metrics.index, y=intuitive_metrics['Predictions within 0.5 magnitude (%)'], palette='viridis')
plt.title('Accuracy: Predictions Within 0.5 Magnitude')
plt.xticks(rotation=45)
plt.ylabel('Percentage (%)')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# Plot Average Prediction Error
plt.figure(figsize=(10, 6))
sns.barplot(x=intuitive_metrics.index, y=intuitive_metrics['Average prediction error'], palette='viridis')
plt.title('Average Prediction Error')
plt.xticks(rotation=45)
plt.ylabel('Magnitude Difference')
plt.xlabel('Model')
plt.tight_layout()
plt.show()

# ===============================
# Prediction Accuracy Breakdown by Error Categories
# ===============================

error_categories = {
    'Excellent (≤0.3)': 0.3,
    'Good (≤0.5)': 0.5,
    'Fair (≤1.0)': 1.0,
    'Poor (>1.0)': float('inf')
}

accuracy_breakdown = pd.DataFrame({
    name: {
        category: np.mean(abs(y_test - results[name]['Predictions']) <= threshold) * 100
        for category, threshold in error_categories.items()
    }
    for name in models.keys()
}).T

# Stacked Bar Plot for Error Categories
plt.figure(figsize=(12, 6))
accuracy_breakdown.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')
plt.title('Prediction Accuracy Breakdown by Model')
plt.xlabel('Model')
plt.ylabel('Percentage of Predictions (%)')
plt.legend(title='Prediction Accuracy', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# Prediction Accuracy by Magnitude Range
# ===============================

for name, model in models.items():
    y_pred = results[name]['Predictions']
    
    magnitude_ranges = {
        'Minor (< 2.0)': (0, 2.0),
        'Light (2.0-3.0)': (2.0, 3.0),
        'Moderate (3.0-4.0)': (3.0, 4.0),
        'Strong (> 4.0)': (4.0, float('inf'))
    }
    
    accuracy_by_range = {}
    for range_name, (min_mag, max_mag) in magnitude_ranges.items():
        mask = (y_test >= min_mag) & (y_test < max_mag)
        if mask.any():
            accuracy = np.mean(abs(y_test[mask] - y_pred[mask]) <= 0.5) * 100
            accuracy_by_range[range_name] = accuracy
    
    results[name].update({'Accuracy_by_Range': accuracy_by_range})

plt.figure(figsize=(10, 6))  
accuracy_data = pd.DataFrame({name: results[name]['Accuracy_by_Range'] 
                              for name in models.keys()}).T

accuracy_data.plot(kind='bar', ax=plt.gca(), colormap='viridis')
plt.title('Prediction Accuracy by Magnitude Range')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.legend(title='Magnitude Range', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
