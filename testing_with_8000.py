import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor

# 1️⃣ Load Data
data = pd.read_csv('cleaned_earthquake_catalogue.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

# Define features
features = ['Timestamp', 'Latitude', 'Longitude', 'Depth']

# 2️⃣ Stratified Sampling
ranges = {
    'Minor': (0, 2.0),
    'Light': (2.0, 3.0),
    'Moderate': (3.0, 4.0),
    'Strong': (4.0, np.inf)
}

samples = []
for label, (min_mag, max_mag) in ranges.items():
    subset = data[(data['Magnitude'] >= min_mag) & (data['Magnitude'] < max_mag)]
    samples.append(subset.sample(n=8000, random_state=42))

balanced_data = pd.concat(samples).sort_values(by='Datetime').reset_index(drop=True)

# Split Features and Target
x = balanced_data[features]
y = balanced_data['Magnitude']

# 3️⃣ Chronological Train/Test Split
split_idx = int(len(balanced_data) * 0.8)
x_train = x.iloc[:split_idx]
y_train = y.iloc[:split_idx]
x_test = x.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# 4️⃣ Scale After Split
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5️⃣ Neural Network Definition
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

# 6️⃣ Define Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'SVR': SVR(kernel='rbf', C=1.0),
    'KNN': KNeighborsRegressor(n_neighbors=3),
    'Neural Network': KerasRegressor(model=create_nn_model, epochs=50, batch_size=32, verbose=0)
}

# 7️⃣ Train and Evaluate Models
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

# 📊 Plot Metrics
metrics_df = pd.DataFrame({
    name: {'MSE': results[name]['MSE'], 'MAE': results[name]['MAE'], 'R2': results[name]['R2']}
    for name in models.keys()
}).T

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.barplot(x=metrics_df.index, y=metrics_df['MSE'], palette='viridis')
plt.title('Mean Squared Error by Model')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.barplot(x=metrics_df.index, y=metrics_df['MAE'], palette='viridis')
plt.title('Mean Absolute Error by Model')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.barplot(x=metrics_df.index, y=metrics_df['R2'], palette='viridis')
plt.title('R² Score by Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📈 Additional Analysis
for name, model in models.items():
    y_pred = results[name]['Predictions']
    within_0_5 = np.mean(abs(y_test - y_pred) <= 0.5) * 100
    avg_diff = np.mean(abs(y_test - y_pred))
    results[name].update({'Within_0.5': within_0_5, 'Avg_Diff': avg_diff})

intuitive_metrics = pd.DataFrame({
    name: {
        'Predictions within 0.5 magnitude (%)': results[name]['Within_0.5'],
        'Average prediction error': results[name]['Avg_Diff'],
    } for name in models.keys()
}).T

# 📊 Plot Predictions Within 0.5 Magnitude
plt.figure(figsize=(10, 6))
sns.barplot(x=intuitive_metrics.index, y=intuitive_metrics['Predictions within 0.5 magnitude (%)'], palette='viridis')
plt.title('Predictions Within ±0.5 Magnitude')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📊 Plot Average Prediction Error
plt.figure(figsize=(10, 6))
sns.barplot(x=intuitive_metrics.index, y=intuitive_metrics['Average prediction error'], palette='viridis')
plt.title('Average Prediction Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📏 Accuracy by Magnitude Range
for name, model in models.items():
    y_pred = results[name]['Predictions']
    magnitude_ranges = {
        'Minor (< 2.0)': (0, 2.0),
        'Light (2.0 - 3.0)': (2.0, 3.0),
        'Moderate (3.0 - 4.0)': (3.0, 4.0),
        'Strong (≥ 4.0)': (4.0, np.inf)
    }
    
    accuracy_by_range = {}
    for range_name, (min_mag, max_mag) in magnitude_ranges.items():
        mask = (y_test >= min_mag) & (y_test < max_mag)
        if mask.any():
            accuracy = np.mean(abs(y_test[mask] - y_pred[mask]) <= 0.5) * 100
            accuracy_by_range[range_name] = accuracy
    
    results[name].update({'Accuracy_by_Range': accuracy_by_range})

plt.figure(figsize=(10, 6))  
accuracy_data = pd.DataFrame({name: results[name]['Accuracy_by_Range'] for name in models.keys()}).T
accuracy_data.plot(kind='bar', colormap='viridis')
plt.title('Prediction Accuracy by Magnitude Range')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.legend(title='Magnitude Range', bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
