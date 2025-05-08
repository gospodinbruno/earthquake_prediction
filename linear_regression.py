import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

def run_linear_regression(data):
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
    model = LinearRegression(
        fit_intercept=True,
        n_jobs=-1
    )
    
    print("Training Linear Regression model...")
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    # --- Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📏 Linear Regression Performance:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")

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

    # --- Plot 4: Residuals vs Predicted ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Magnitude')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot 5: Distribution of True vs Predicted ---
    plt.figure(figsize=(8, 5))
    sns.histplot(y_test, color='blue', label='True', kde=True, stat='density', bins=30, alpha=0.5)
    sns.histplot(y_pred, color='orange', label='Predicted', kde=True, stat='density', bins=30, alpha=0.5)
    plt.title("Predicted vs True Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Plot 6: Coefficients/Feature Importance ---
    coef = pd.Series(model.coef_, index=features)
    plt.figure(figsize=(8, 5))
    coef.sort_values().plot(kind='barh', color='purple')
    plt.title('Linear Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # --- Plot 7: Permutation Importance ---
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
    
    # --- Plot 9: Error by Magnitude Range ---
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
    
    # --- Plot 10: LR-specific: Simple vs Polynomial Features ---
    degrees = [1, 2, 3]
    models_mae = []
    models_r2 = []
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_train_poly = poly.fit_transform(x_train_scaled)
        x_test_poly = poly.transform(x_test_scaled)
        
        # Fit a model
        poly_model = LinearRegression()
        poly_model.fit(x_train_poly, y_train)
        poly_pred = poly_model.predict(x_test_poly)
        
        # Calculate performance metrics
        poly_mae = mean_absolute_error(y_test, poly_pred)
        poly_r2 = r2_score(y_test, poly_pred)
        
        models_mae.append(poly_mae)
        models_r2.append(poly_r2)
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Linear', 'Quadratic', 'Cubic'], models_mae, color='maroon')
    plt.title('Effect of Polynomial Degree on MAE')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Linear', 'Quadratic', 'Cubic'], models_r2, color='navy')
    plt.title('Effect of Polynomial Degree on R²')
    plt.ylabel('R² Score')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # --- Plot 11: Geographic Distribution of Errors ---
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
    
    # --- Plot 12: QQ Plot for Residuals (to check for normality) ---
    plt.figure(figsize=(8, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('QQ Plot of Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the script as standalone program
if __name__ == "__main__":
    try:
        # Load the earthquake data
        data = pd.read_csv("cleaned_earthquake_catalogue.csv")
        print(f"Loaded dataset with {len(data)} records")
        
        # Run the Linear Regression analysis
        run_linear_regression(data)
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the cleaned_earthquake_catalogue.csv file exists in the current directory") 