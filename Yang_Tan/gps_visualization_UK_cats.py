import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset (ensure the file path is correct)
file_path = "archive/Pet Cats United States.csv"  # Adjust this path to your dataset location
df = pd.read_csv(file_path)

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the DataFrame for only the date "2013-08-14"
df_filtered = df[df['timestamp'].dt.date == pd.to_datetime('2013-08-14').date()]

# Select relevant columns: individual-local-identifier, location-long, location-lat, timestamp
df_extracted = df_filtered[["individual-local-identifier", "location-lat", "location-long", "timestamp"]].copy()

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_extracted,
    geometry=gpd.points_from_xy(df_extracted["location-long"], df_extracted["location-lat"]),
    crs="EPSG:4326"
)

# Sort by individual-local-identifier and timestamp
gdf = gdf.sort_values(by=["individual-local-identifier", "timestamp"])

# Create edges (lines) between consecutive points for the same individual-local-identifier
lines = []

# Iterate through each individual-local-identifier group
for _, group in gdf.groupby('individual-local-identifier'):
    for i in range(len(group) - 1):  # Connect consecutive points
        point1 = group.iloc[i].geometry
        point2 = group.iloc[i + 1].geometry
        lines.append(LineString([point1, point2]))

# Create GeoDataFrame for edges (lines between consecutive locations of the same individual)
edges_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

# Load the USA shapefile for boundaries (replace this with your shapefile)
usa_boundaries = gpd.read_file("cb_2022_us_nation_20m.shp")
print(usa_boundaries)
# Filter the USA shapefile to the USA (if necessary)
usa = usa_boundaries['geometry']

# Plot the map for the USA
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the country boundary (USA) on the map
usa.plot(ax=ax, color='lightgray', edgecolor='black')

# Plot the cats' locations as red points
gdf.plot(ax=ax, color="red", label="Cats' Locations", markersize=5)

# Plot the edges (lines connecting consecutive points of the same individual)
edges_gdf.plot(ax=ax, color="blue", linewidth=0.5, label="Movement Edges")

# Add a legend and title
plt.legend()
plt.title("Cats' Locations and Movement Edges in the USA on 2013-08-14")
plt.show()