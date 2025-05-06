import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def run_knn_regression(data):
    # --- Feature preparation ---
    athens_center_lat = 37.9838
    athens_center_lon = 23.7275
    data['lat_diff'] = data['Latitude'] - athens_center_lat
    data['lon_diff'] = data['Longitude'] - athens_center_lon
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
    data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

    features = ['Timestamp', 'Latitude', 'Longitude', 'Depth']
    x = data[features]
    y = data['Magnitude']

    # --- Feature Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    corr = data[features + ['Magnitude']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # --- Split ---
    split_idx = int(len(data) * 0.8)
    x_train = x.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    x_test = x.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # --- Scaling ---
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # --- Model ---
    model = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        p=2,  # Euclidean distance
        n_jobs=-1
    )
    
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    # --- Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"🏠 kNN Regression Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

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

    # --- Plot 4: Distribution of True vs Predicted ---
    plt.figure(figsize=(8, 5))
    sns.histplot(y_test, color='blue', label='True', kde=True, stat='density', bins=30, alpha=0.5)
    sns.histplot(y_pred, color='orange', label='Predicted', kde=True, stat='density', bins=30, alpha=0.5)
    plt.title("Predicted vs True Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Plot 5: Permutation Importance ---
    try:
        perm_importance = permutation_importance(model, x_test_scaled, y_test, 
                                               n_repeats=10, random_state=42)
        perm_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
        plt.title('Feature Importance (Permutation Method)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not calculate permutation importance: {e}")

    # --- Plot 6: Error by Depth Bin ---
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
    
    # --- Plot 7: Error by Magnitude Range ---
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
    
    # --- Plot 8: kNN-specific: Effect of number of neighbors ---
    n_neighbors_range = range(1, 21)
    mae_scores = []
    r2_scores = []
    
    for n in n_neighbors_range:
        temp_model = KNeighborsRegressor(n_neighbors=n, weights='distance')
        temp_model.fit(x_train_scaled, y_train)
        temp_pred = temp_model.predict(x_test_scaled)
        mae_scores.append(mean_absolute_error(y_test, temp_pred))
        r2_scores.append(r2_score(y_test, temp_pred))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(n_neighbors_range, mae_scores, 'o-', color='blue')
    plt.title('MAE vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(n_neighbors_range, r2_scores, 'o-', color='green')
    plt.title('R² vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('R² Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # --- Plot 9: Geographic Distribution of Errors ---
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

# Run the script as standalone program
if __name__ == "__main__":
    try:
        # Load the earthquake data
        data = pd.read_csv("cleaned_earthquake_catalogue.csv")
        print(f"Loaded dataset with {len(data)} records")
        
        # Run the kNN regression analysis
        run_knn_regression(data)
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the cleaned_earthquake_catalogue.csv file exists in the current directory") 