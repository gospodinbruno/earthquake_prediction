import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


# Load the dataset
data = pd.read_csv('cleaned_earthquake_catalogue.csv')


athens_center_lat = 37.9838
athens_center_lon = 23.7275

# Add relative distance features
data['lat_diff'] = data['Latitude'] - athens_center_lat
data['lon_diff'] = data['Longitude'] - athens_center_lon

data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y-%m-%d')
data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
print(data['Timestamp'].head) 
features = ['Timestamp','Latitude', 'Longitude', 'Depth']

x = data[features]  # features like Timestamp, Latitude, etc.
y = data['Magnitude']

# Manual time split
split_idx = int(len(data) * 0.8)  # 80% for training

x_train = x.iloc[:split_idx]
y_train = y.iloc[:split_idx]

x_test = x.iloc[split_idx:]
y_test = y.iloc[split_idx:]


plt.hist(y, bins=30, edgecolor='black')
plt.title('Histogram of Earthquake Magnitudes (Athens Only)')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




# Define models with hyperparameters to search
param_grid = {
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 5]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance']
        }
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'Neural Network': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'solver': ['adam']
        }
    }
}

# Store results
results = []

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#Train and evaluate each model
for name, mp in param_grid.items():
    grid = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='r2', n_jobs=-1)
    grid.fit(x_train_scaled, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(x_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2,
        'Best Params': grid.best_params_
    })

# Convert to DataFrame
results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
print("\n🏆 Tuned Model Leaderboard (sorted by R² Score):\n")
print(results_df[['Model', 'MAE', 'RMSE', 'R2 Score', 'Best Params']].to_string(index=False))


# Assuming your Athens magnitudes are in a list/array called 'magnitudes'

