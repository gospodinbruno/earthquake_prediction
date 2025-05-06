import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("Loading and preparing data...")

# Load the dataset
df = pd.read_csv('earthquake.csv')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Handle datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Hour'] = df['Datetime'].dt.hour

# Create time-based features
df['Timestamp'] = (df['Datetime'] - pd.Timestamp("1964-01-01")) / pd.Timedelta(seconds=1)
df['TimeSinceLast'] = df.groupby(['Latitude', 'Longitude'])['Timestamp'].diff()

# Fill NaN values for first earthquakes in each location
df['TimeSinceLast'] = df['TimeSinceLast'].fillna(df['TimeSinceLast'].median())

# Calculate distance from reference point (Athens)
athens_center_lat = 37.9838
athens_center_lon = 23.7275

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Add distance features
df['DistanceFromAthens'] = haversine_distance(
    df['Latitude'], df['Longitude'], 
    athens_center_lat, athens_center_lon
)

# Add relative position features
df['lat_diff'] = df['Latitude'] - athens_center_lat
df['lon_diff'] = df['Longitude'] - athens_center_lon

# Calculate rolling statistics per region
# Define grid regions
lat_bins = pd.cut(df['Latitude'], bins=10)
lon_bins = pd.cut(df['Longitude'], bins=10)
df['Region'] = pd.factorize(lat_bins.astype(str) + lon_bins.astype(str))[0]

# Calculate rolling averages for each region
df['RegionMeanMag'] = df.groupby('Region')['Magnitude'].transform(
    lambda x: x.expanding().mean().shift()
)
df['RegionMaxMag'] = df.groupby('Region')['Magnitude'].transform(
    lambda x: x.expanding().max().shift()
)
df['RegionStdMag'] = df.groupby('Region')['Magnitude'].transform(
    lambda x: x.expanding().std().shift()
)

# Fill NA values for first earthquakes in regions
df['RegionMeanMag'] = df['RegionMeanMag'].fillna(df['Magnitude'].mean())
df['RegionMaxMag'] = df['RegionMaxMag'].fillna(df['Magnitude'].mean())
df['RegionStdMag'] = df['RegionStdMag'].fillna(df['Magnitude'].std())

# Create time-based features - avoid duplicate datetime issue by creating a unique index
print("Creating time-based rolling features...")
for window in [5, 10, 20]:
    # For each region, calculate rolling features
    region_dfs = []
    
    for region in df['Region'].unique():
        # Get data for this region and sort by datetime
        region_df = df[df['Region'] == region].sort_values('Datetime').copy()
        
        # Create a unique datetime index by adding microseconds if needed
        if region_df['Datetime'].duplicated().any():
            for i, (idx, row) in enumerate(region_df[region_df['Datetime'].duplicated()].iterrows()):
                region_df.loc[idx, 'Datetime'] = row['Datetime'] + pd.Timedelta(microseconds=i+1)
        
        # Set index to datetime (now unique)
        region_df.set_index('Datetime', inplace=True)
        
        # Calculate rolling features
        region_df[f'Count_{window}d'] = region_df['Magnitude'].rolling(f'{window}D').count()
        region_df[f'Mean_{window}d'] = region_df['Magnitude'].rolling(f'{window}D').mean()
        region_df[f'Max_{window}d'] = region_df['Magnitude'].rolling(f'{window}D').max()
        
        # Reset index to get Datetime back as a column
        region_df.reset_index(inplace=True)
        
        # Add to list of processed regions
        region_dfs.append(region_df)
    
    # Combine all regions back into one dataframe
    df_new = pd.concat(region_dfs)
    
    # Sort back to original order
    df_new = df_new.sort_index()
    
    # Add the new columns to our original dataframe
    df[f'Count_{window}d'] = df_new[f'Count_{window}d'].values
    df[f'Mean_{window}d'] = df_new[f'Mean_{window}d'].values
    df[f'Max_{window}d'] = df_new[f'Max_{window}d'].values

# Fill NaN values
print("Filling missing values...")
for col in df.columns[df.isna().any()]:
    if col != 'Datetime':
        df[col] = df[col].fillna(df[col].median())

# Create a feature correlation heatmap
print("Generating correlation heatmap...")
plt.figure(figsize=(12, 10))
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation = df[numerical_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Exploratory Data Analysis
print("\nPerforming exploratory data analysis...")

plt.figure(figsize=(10, 6))
sns.histplot(df['Magnitude'], bins=30, kde=True)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.savefig('magnitude_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
df.groupby('Year')['Magnitude'].mean().plot(kind='line', marker='o')
plt.title('Average Earthquake Magnitude by Year')
plt.xlabel('Year')
plt.ylabel('Average Magnitude')
plt.grid(True)
plt.savefig('magnitude_by_year.png')
plt.close()

plt.figure(figsize=(12, 10))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Magnitude'], 
            cmap='viridis', alpha=0.6, edgecolors='k', s=df['Magnitude']*20)
plt.colorbar(label='Magnitude')
plt.title('Geographical Distribution of Earthquakes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig('earthquake_map.png')
plt.close()

# Define features and target
print("\nPreparing features for modeling...")

# Define features for modeling
features = [
    'Latitude', 'Longitude', 'Depth', 'Year', 'Month', 'Day', 'Hour',
    'DayOfWeek', 'TimeSinceLast', 'DistanceFromAthens', 'lat_diff', 'lon_diff',
    'RegionMeanMag', 'RegionMaxMag', 'RegionStdMag', 'Count_5d', 'Mean_5d', 
    'Max_5d', 'Count_10d', 'Mean_10d', 'Max_10d', 'Count_20d', 'Mean_20d', 'Max_20d'
]

X = df[features]
y = df['Magnitude']

# Use time-based split instead of random split
# Sort by datetime to ensure chronological order
df = df.sort_values('Datetime')
X = df[features]
y = df['Magnitude']

# Use the last 20% of the data for testing (chronological split)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Support Vector Machine': SVR(kernel='rbf')
}

# Train and evaluate each model
results = []

print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2
    })
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'{name}: Actual vs Predicted Magnitudes')
    plt.grid(True)
    plt.savefig(f'{name.replace(" ", "_").lower()}_predictions.png')
    plt.close()

# Convert results to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R2 Score', ascending=False)

# Display leaderboard
print("\n🏆 Model Leaderboard (sorted by R² Score):\n")
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning on the best model...")

# Determine the best model from results
best_model_name = results_df.iloc[0]['Model']
print(f"Best model: {best_model_name}")

# Define parameter grids for different models
param_grids = {
    'Linear Regression': {},  # Linear regression doesn't have hyperparameters to tune
    
    'Decision Tree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'gamma': [0, 0.1, 0.2]
    },
    
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
    },
    
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly']
    }
}

# Get parameter grid for best model
if best_model_name in param_grids and param_grids[best_model_name]:
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create model instance
    best_model = models[best_model_name]
    
    # Set up grid search
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grids[best_model_name],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best cross-validation score (neg MSE): {best_score}")
    
    # Train the model with best parameters
    tuned_model = grid_search.best_estimator_
    tuned_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_tuned = tuned_model.predict(X_test_scaled)
    
    # Calculate metrics
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    print("\nTuned model performance:")
    print(f"MAE: {mae_tuned:.4f}")
    print(f"RMSE: {rmse_tuned:.4f}")
    print(f"R² Score: {r2_tuned:.4f}")
    
    # Plot actual vs predicted for tuned model
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_tuned, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title(f'Tuned {best_model_name}: Actual vs Predicted Magnitudes')
    plt.grid(True)
    plt.savefig('tuned_model_predictions.png')
    plt.close()

# Feature importance for tree-based models
if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
    print("\nCalculating feature importance...")
    
    # Get the best model (either tuned or original)
    final_model = tuned_model if 'tuned_model' in locals() else models[best_model_name]
    
    # Get feature importances
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances ({best_model_name})')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.close()
        
        # Print feature importances
        print("\nFeature Importances:")
        for i in indices:
            print(f"{features[i]}: {importances[i]:.4f}")

# Test model on future predictions
print("\nTesting forecasting capability...")

# Let's create a function to make predictions for future earthquakes
def predict_future_earthquake(location_lat, location_lon, depth, date, region_data=None):
    """
    Predict earthquake magnitude for a future earthquake at a given location and time.
    
    Parameters:
    - location_lat: Latitude of the location
    - location_lon: Longitude of the location
    - depth: Depth of the earthquake
    - date: Date for the prediction (datetime object)
    - region_data: Dictionary with region statistics (optional)
    
    Returns:
    - Predicted magnitude
    """
    # Create a DataFrame with a single row for the prediction
    future_eq = pd.DataFrame({
        'Latitude': [location_lat],
        'Longitude': [location_lon],
        'Depth': [depth],
        'Year': [date.year],
        'Month': [date.month],
        'Day': [date.day],
        'Hour': [date.hour],
        'DayOfWeek': [date.weekday()],
    })
    
    # Calculate distance from Athens
    future_eq['DistanceFromAthens'] = haversine_distance(
        future_eq['Latitude'], future_eq['Longitude'], 
        athens_center_lat, athens_center_lon
    )
    
    # Add relative position features
    future_eq['lat_diff'] = future_eq['Latitude'] - athens_center_lat
    future_eq['lon_diff'] = future_eq['Longitude'] - athens_center_lon
    
    # If region data is provided, use it
    if region_data:
        future_eq['RegionMeanMag'] = region_data.get('mean_mag', df['Magnitude'].mean())
        future_eq['RegionMaxMag'] = region_data.get('max_mag', df['Magnitude'].max())
        future_eq['RegionStdMag'] = region_data.get('std_mag', df['Magnitude'].std())
        future_eq['Count_5d'] = region_data.get('count_5d', df['Count_5d'].median())
        future_eq['Mean_5d'] = region_data.get('mean_5d', df['Mean_5d'].median())
        future_eq['Max_5d'] = region_data.get('max_5d', df['Max_5d'].median())
        future_eq['Count_10d'] = region_data.get('count_10d', df['Count_10d'].median())
        future_eq['Mean_10d'] = region_data.get('mean_10d', df['Mean_10d'].median())
        future_eq['Max_10d'] = region_data.get('max_10d', df['Max_10d'].median())
        future_eq['Count_20d'] = region_data.get('count_20d', df['Count_20d'].median())
        future_eq['Mean_20d'] = region_data.get('mean_20d', df['Mean_20d'].median())
        future_eq['Max_20d'] = region_data.get('max_20d', df['Max_20d'].median())
    else:
        # Use median values from the dataset
        for col in ['RegionMeanMag', 'RegionMaxMag', 'RegionStdMag', 
                   'Count_5d', 'Mean_5d', 'Max_5d', 
                   'Count_10d', 'Mean_10d', 'Max_10d',
                   'Count_20d', 'Mean_20d', 'Max_20d']:
            future_eq[col] = df[col].median()
    
    # Use median for time since last
    future_eq['TimeSinceLast'] = df['TimeSinceLast'].median()
    
    # Make sure all required features are present
    missing_cols = [col for col in features if col not in future_eq.columns]
    for col in missing_cols:
        future_eq[col] = df[col].median()
    
    # Reorder columns to match training data
    future_eq = future_eq[features]
    
    # Scale features
    future_eq_scaled = scaler.transform(future_eq)
    
    # Get the best model (either tuned or original)
    final_model = tuned_model if 'tuned_model' in locals() else models[best_model_name]
    
    # Make prediction
    predicted_magnitude = final_model.predict(future_eq_scaled)[0]
    
    return predicted_magnitude

# Example usage
print("\nExample predictions for future earthquakes:")

# Predict for Athens area
predicted_magnitude = predict_future_earthquake(
    location_lat=37.9838,
    location_lon=23.7275,
    depth=10,
    date=datetime.datetime.now() + datetime.timedelta(days=30)
)
print(f"Predicted magnitude for Athens area in 30 days: {predicted_magnitude:.2f}")

# Predict for a high-risk area (based on historical data)
high_risk_lat = df.loc[df['Magnitude'].idxmax(), 'Latitude']
high_risk_lon = df.loc[df['Magnitude'].idxmax(), 'Longitude']

predicted_magnitude = predict_future_earthquake(
    location_lat=high_risk_lat,
    location_lon=high_risk_lon,
    depth=10,
    date=datetime.datetime.now() + datetime.timedelta(days=30)
)
print(f"Predicted magnitude for high-risk area in 30 days: {predicted_magnitude:.2f}")

print("\nEarthquake prediction analysis complete!")
print("All results and visualizations have been saved to the current directory.") 