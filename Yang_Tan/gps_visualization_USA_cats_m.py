import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from folium.plugins import MarkerCluster

# Load the dataset (ensure the file path is correct)
file_path = "archive/Pet Cats United States.csv"  # Adjust this path to your dataset location
df = pd.read_csv(file_path)

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the DataFrame for only the morning of 2013-08-14 (00:00 to 12:00) and the specific individual-local-identifier "Motorcat"
start_time = pd.to_datetime("2013-08-14 12:00:00")
end_time = pd.to_datetime("2016-06-21 23:59:00")
df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time) & (df['individual-local-identifier'] == "Motorcat")]

# Select relevant columns
df_extracted = df_filtered[["individual-local-identifier", "location-lat", "location-long", "timestamp"]].copy()

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_extracted,
    geometry=gpd.points_from_xy(df_extracted["location-long"], df_extracted["location-lat"]),
    crs="EPSG:4326"  # WGS 84
)

# Sort by timestamp
gdf = gdf.sort_values(by=["timestamp"])

# Create the folium map centered around the first location of the motorcattag cat (using its first lat/lon)
first_lat = gdf.iloc[0]["location-lat"]
first_lon = gdf.iloc[0]["location-long"]
m = folium.Map(location=[first_lat, first_lon], zoom_start=14)

# Plot the movement path as a single polyline (without individual dots)
path_coordinates = [[row["location-lat"], row["location-long"]] for _, row in gdf.iterrows()]
folium.PolyLine(path_coordinates, color="blue", weight=2, opacity=0.7).add_to(m)

# Save this specific map to a new HTML file (overwriting any existing file with the same name)
m.save("motorcattag_location_path_2013_08_14.html")