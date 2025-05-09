import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# --- Load Data ---
data = pd.read_csv("cleaned_earthquake_catalogue.csv")
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

features = ['Timestamp', 'Latitude', 'Longitude', 'Depth']
x = data[features]
y = data['Magnitude']

# --- Scale Features ---
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# --- Chronological Split ---
split_idx = int(len(x) * 0.8)
x_train, x_test = x_scaled[:split_idx], x_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Neural Network Definition ---
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse'])
    return model

model = create_nn_model(x_train.shape[1])

# --- Training ---
history = model.fit(
    x_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(x_test, y_test), 
    verbose=1
)

# --- Predictions ---
y_pred = model.predict(x_test).flatten()

# --- Evaluation ---
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"🧠 Neural Network Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

# --- Plot Training & Validation Loss ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['mse'], label='Training Loss')
plt.plot(history.history['val_mse'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Add any additional plots below if needed...
# --- Plot 1: True vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('True vs Predicted Magnitudes')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Residual Histogram ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.title('Residuals Histogram')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Residuals vs True ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Magnitude')
plt.ylabel('Residual')
plt.title('Residuals vs True Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Residuals Over Time ---
plt.figure(figsize=(12, 5))
plt.plot(data['Datetime'].iloc[split_idx:], residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Prediction Error Over Time')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(y_test, color='blue', label='True', kde=True, stat='density', bins=30, alpha=0.5)
sns.histplot(y_pred, color='orange', label='Predicted', kde=True, stat='density', bins=30, alpha=0.5)
plt.title("Predicted vs True Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 8: Error by Depth Bin ---
depth_bins = pd.cut(data['Depth'].iloc[split_idx:], bins=10)
error_by_depth = pd.DataFrame({'residual': residuals, 'bin': depth_bins})
grouped = error_by_depth.groupby('bin')['residual'].agg(['mean', 'std'])

plt.figure(figsize=(10, 5))
grouped['mean'].plot(kind='bar', yerr=grouped['std'], capsize=4, color='coral')
plt.title('Mean Residual by Depth Bin')
plt.ylabel('Mean Residual')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot: Error by Magnitude Range ---
mag_bins = pd.cut(y_test, bins=5)
error_by_mag = pd.DataFrame({'error': np.abs(residuals), 'bin': mag_bins})
grouped_mag = error_by_mag.groupby('bin')['error'].agg(['mean', 'std'])

plt.figure(figsize=(10, 5))
grouped_mag['mean'].plot(kind='bar', yerr=grouped_mag['std'], capsize=4, color='skyblue')
plt.title('Mean Absolute Error by Magnitude Range')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot: Geographic Distribution of Errors ---
plt.figure(figsize=(10, 8))
sc = plt.scatter(data['Longitude'].iloc[split_idx:], data['Latitude'].iloc[split_idx:], 
                c=np.abs(residuals), cmap='Reds', alpha=0.7, s=50)
plt.colorbar(sc, label='Absolute Error')
plt.title('Geographic Distribution of Prediction Errors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()
