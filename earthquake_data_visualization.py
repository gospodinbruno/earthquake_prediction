import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Load data
data = pd.read_csv('cleaned_earthquake_catalogue.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Add useful features
data['Year'] = data['Datetime'].dt.year
data['Month'] = data['Datetime'].dt.month
data['DayOfWeek'] = data['Datetime'].dt.dayofweek
data['Hour'] = data['Datetime'].dt.hour


data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)

# Filter: Magnitudes >= 2.0
data = data[data['Magnitude'] >= 2.0].reset_index(drop=True)

# --- Plot 1: Magnitude histogram ---
plt.figure(figsize=(10, 6))
plt.hist(data['Magnitude'], bins=30, edgecolor='black')
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Earthquakes by year ---
plt.figure(figsize=(12, 6))
data.groupby('Year').size().plot(kind='bar', color='steelblue')
plt.title('Earthquake Count by Year')
plt.ylabel('Number of Earthquakes')
plt.xlabel('Year')
plt.tight_layout()
plt.show()

# --- Plot 3: Depth vs Magnitude ---
plt.figure(figsize=(10, 6))
plt.scatter(data['Depth'], data['Magnitude'], alpha=0.4, s=5)
plt.title('Depth vs Magnitude')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Magnitude over time ---
plt.figure(figsize=(14, 5))
plt.plot(data['Datetime'], data['Magnitude'], alpha=0.4, linewidth=0.5)
plt.title('Magnitude of Earthquakes Over Time')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# --- Plot 5: Correlation heatmap ---
plt.figure(figsize=(8, 6))
corr = data[['Latitude', 'Longitude', 'Depth', 'Magnitude']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.show()

# --- Plot 6: Epicenter heatmap (Athens) with map background ---
plt.figure(figsize=(12, 10))

# Athens region coordinates (approximate)
athens_lon, athens_lat = 23.7275, 37.9838
lon_min, lon_max = 22.5, 25.0  # Longitude bounds for Athens region
lat_min, lat_max = 37.0, 39.0  # Latitude bounds for Athens region

# Filter data to Athens region
athens_data = data[(data['Longitude'] >= lon_min) & (data['Longitude'] <= lon_max) & 
                   (data['Latitude'] >= lat_min) & (data['Latitude'] <= lat_max)]

# Create scatter plot with point size and color based on magnitude
scatter = plt.scatter(athens_data['Longitude'], athens_data['Latitude'], 
                     c=athens_data['Magnitude'], cmap='YlOrRd', 
                     alpha=0.7, s=athens_data['Magnitude']**2, 
                     edgecolor='k', linewidth=0.3)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Magnitude')

# Add Athens marker
plt.plot(athens_lon, athens_lat, 'b*', markersize=15, label='Athens')

# Set plot limits to Athens region
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)

# Add grid
plt.grid(True, linestyle='--', alpha=0.6)

# Add title and labels
plt.title('Earthquake Epicenters in Athens Region', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

try:
    # Try to add basemap using contextily
    import contextily as ctx
    
    # Convert axes to web mercator projection
    ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
    
    plt.title('Earthquake Epicenters with Map Background (Athens Region)', fontsize=16)
except Exception as e:
    print(f"Couldn't add map background: {e}")
    
plt.tight_layout()
plt.show()

# --- Plot 7: Average magnitude by month ---
plt.figure(figsize=(10, 5))
data.groupby('Month')['Magnitude'].mean().plot(kind='bar', color='darkorange')
plt.title('Average Earthquake Magnitude by Month')
plt.xlabel('Month')
plt.ylabel('Avg Magnitude')
plt.tight_layout()
plt.show()

# --- Plot 8: Earthquakes per weekday ---
plt.figure(figsize=(8, 5))
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.countplot(x=data['DayOfWeek'], palette='viridis')
plt.xticks(ticks=range(7), labels=day_labels)
plt.title('Earthquake Frequency by Day of Week')
plt.xlabel('Day')
plt.ylabel('Count')
plt.tight_layout()
plt.show()




