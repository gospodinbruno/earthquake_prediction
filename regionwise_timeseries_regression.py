import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------
# 1. Load & Prepare Data
# ---------------------
data = pd.read_csv('cleaned_earthquake_catalogue.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values('Datetime')
data = data[data['Magnitude'] >= 2.0].reset_index(drop=True)

# ---------------------
# 2. Cluster by Location (lat, lon → 5 spatial regions)
# ---------------------
coords = data[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=5, random_state=42).fit(coords)
data['region'] = kmeans.labels_

# ---------------------
# 3. Generate Lag Features PER REGION
# ---------------------
def create_lagged_region_df(region_df):
    region_df = region_df.sort_values('Datetime').reset_index(drop=True)
    region_df['mag_t-1'] = region_df['Magnitude'].shift(1)
    region_df['mag_t-2'] = region_df['Magnitude'].shift(2)
    return region_df.dropna()

region_results = []
region_models = {}

for region_id in sorted(data['region'].unique()):
    region_df = data[data['region'] == region_id].copy()
    region_df = create_lagged_region_df(region_df)

    X = region_df[['mag_t-1', 'mag_t-2']]
    y = region_df['Magnitude']

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Optional: scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = r2_score(y_test, y_pred)

    region_results.append({
        'Region': region_id,
        'Sample Size': len(region_df),
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

    region_models[region_id] = (model, scaler)

    # Plot: True vs Pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Region {region_id} | True vs Predicted Magnitude')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'region_{region_id}_true_vs_pred.png')
    plt.close()

# ---------------------
# 4. Show Performance Summary
# ---------------------
results_df = pd.DataFrame(region_results).sort_values(by='R²', ascending=False)
print("\n📊 Region-wise Time Series Regression Performance:\n")
print(results_df.to_string(index=False))
