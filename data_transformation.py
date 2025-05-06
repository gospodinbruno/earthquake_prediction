import pandas as pd

# Read all lines
with open('earthquake_prediction.txt', 'r') as file:
    lines = file.readlines()

data = []

for line in lines:
    # Skip header lines or empty lines
    if not line.strip() or line.startswith('DATE') or line.startswith('('):
        continue

    try:
        year = line[1:5].strip()
        month = line[6:9].strip()
        day = line[10:12].strip()
        hour = line[15:17].strip()
        minute = line[18:20].strip()
        second = line[21:25].strip()
        latitude = line[26:33].strip()
        longitude = line[34:41].strip()
        depth = line[42:47].strip()
        magnitude = line[55:59].strip()

        # Assemble into one dictionary
        data.append({
            'Year': year,
            'Month': month,
            'Day': day,
            'Hour': hour,
            'Minute': minute,
            'Second': second,
            'Latitude': latitude,
            'Longitude': longitude,
            'Depth_km': depth,
            'Magnitude': magnitude
        })
    except Exception as e:
        print(f"Skipping line because of error: {e}")
        continue

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv('earthquake_catalogue.csv', index=False)

print("Done!")