import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from datetime import datetime
import contextily as ctx
import plotly.express as px
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
                     alpha=0.7, s=10 ** (athens_data['Magnitude'] - 2), 
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

# --- Plot 7: Magnitude and Depth histograms side by side ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.hist(data['Magnitude'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
sns.kdeplot(data['Magnitude'], color='green', lw=1)
plt.title('Earthquake Magnitude Data')
plt.xlabel('Magnitude Scale')
plt.ylabel('Number of Earthquakes')

plt.subplot(1, 2, 2)
plt.hist(data['Depth'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
sns.kdeplot(data['Depth'], color='blue', lw=1)
plt.title('Earthquake Depth Data')
plt.xlabel('Depth (km)')
plt.ylabel('Number of Earthquakes')

plt.tight_layout()
plt.show()

# Load and filter data
data = pd.read_csv("cleaned_earthquake_catalogue.csv")
data['Datetime'] = pd.to_datetime(data['Datetime'])
data['Year'] = data['Datetime'].dt.year

# Filter last 15 years & keep magnitudes >= 2.0
recent = data[(data['Year'] >= datetime.now().year - 15) & (data['Magnitude'] >= 2)].copy()

# Add a new 'Size' column: non-linear scaling for visual impact
recent['Size'] = recent['Magnitude'] ** 3  # or try np.exp(Mag)

# Plot with Plotly
filtered = data[(data['Year'] >= datetime.now().year - 5) & (data['Magnitude'] >= 2.5)]

# Create Plotly Geo scatter
fig = px.scatter_geo(
    filtered,
    lat='Latitude',
    lon='Longitude',
    color='Magnitude',
    color_continuous_scale='Plasma',
    projection='natural earth',
    scope='europe',  # or 'world' for global
    title='Earthquake Locations by Magnitude (Last 5 Years)',
    opacity=0.7,
)

# Style the map
fig.update_geos(
    showland=True, landcolor="rgb(230, 230, 230)",
    showcountries=True, countrycolor="gray",
    showcoastlines=True, coastlinecolor="gray"
)

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title="Magnitude",
        tickfont=dict(size=20),
        titlefont=dict(size=22)
    )
)

# Display
fig.show()


