import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv('cleaned_earthquake_catalogue.csv')

athens_center_lat = 37.9838
athens_center_lon = 23.7275

# Add relative distance features
data['lat_diff'] = data['Latitude'] - athens_center_lat
data['lon_diff'] = data['Longitude'] - athens_center_lon

# Convert datetime to timestamp
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

# Define features - using all available relevant features
features = [ 'Timestamp', 'Latitude', 'Longitude', 'Depth']

x = data[features]
y = data['Magnitude']

# Manual time split - 80% for training
split_idx = int(len(data) * 0.8)
x_train = x.iloc[:split_idx]
y_train = y.iloc[:split_idx]
x_test = x.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Neural Network hyperparameter grid (smaller)
nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Only 3 architectures
    'activation': ['relu'],                            # Only relu
    'alpha': [0.0001, 0.001],                           # 2 alphas
    'learning_rate': ['adaptive'],                      # Only adaptive
    'solver': ['adam'],                                 # Adam only
    'max_iter': [1000]                                  # Single value
}

print("Starting Neural Network grid search... (smaller grid)")

# Grid search for Neural Network
grid = GridSearchCV(
    MLPRegressor(random_state=42, early_stopping=True),  # Added early_stopping
    nn_param_grid,
    cv=2,                  # Only 2 folds instead of 3
    scoring='r2',
    n_jobs=-1,
    verbose=2
)
grid.fit(x_train_scaled, y_train)

# Get best model
best_model = grid.best_estimator_
print(f"\nBest Neural Network Parameters: {grid.best_params_}")

# Evaluate on test set
y_pred = best_model.predict(x_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nNeural Network Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# --------------- PLOTS -----------------

# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Neural Network: Actual vs Predicted Earthquake Magnitudes')
plt.grid(True)
plt.savefig('neural_network_predictions.png')
plt.show()

# 2. Residuals histogram
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.title('Residuals Histogram')
plt.xlabel('Prediction Error (Residual)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('neural_network_residuals.png')
plt.show()

# 3. Prediction Errors over Magnitude (error vs true magnitude)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Magnitude')
plt.ylabel('Prediction Error (Residual)')
plt.title('Prediction Error vs True Magnitude')
plt.grid(True)
plt.savefig('neural_network_error_vs_magnitude.png')
plt.show()

# 4. Feature Importance (rough approximation)
def calculate_feature_importance(model, feature_names):
    importances = np.abs(model.coefs_[0]).sum(axis=1)
    importances = importances / np.sum(importances)
    return pd.Series(importances, index=feature_names)

feature_importance = calculate_feature_importance(best_model, features)
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh')
plt.xlabel('Relative Importance')
plt.title('Feature Importance for Neural Network Model')
plt.grid(True)
plt.tight_layout()
plt.savefig('neural_network_feature_importance.png')
plt.show()
