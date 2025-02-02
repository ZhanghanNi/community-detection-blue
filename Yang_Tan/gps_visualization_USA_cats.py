import folium
import pandas as pd
import geopandas as gpd
import random

# Load the dataset (ensure the file path is correct)
file_path = "archive/Pet Cats United States.csv"  # Adjust this path to your dataset location
df = pd.read_csv(file_path)

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter the DataFrame for the afternoon of 2013-08-14 (12:00 to 23:59)
start_time = pd.to_datetime("2015-06-21 23:59:00")
end_time = pd.to_datetime("2015-12-21 23:59:00")
df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

# Select relevant columns
df_extracted = df_filtered[["individual-local-identifier", "location-lat", "location-long", "timestamp"]].copy()

# Check how many unique cats (individual-local-identifier) are in the filtered data
unique_cats = df_extracted["individual-local-identifier"].nunique()
print(f"Number of unique cats in the selected time range: {unique_cats}")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_extracted,
    geometry=gpd.points_from_xy(df_extracted["location-long"], df_extracted["location-lat"]),
    crs="EPSG:4326"  # WGS 84
)

# Sort by individual-local-identifier and timestamp
gdf = gdf.sort_values(by=["individual-local-identifier", "timestamp"])

# Create the folium map centered around Raleigh (using a known lat/lon for Raleigh)
m = folium.Map(location=[35.7796, -78.6382], zoom_start=12)

# Generate a consistent random color for each individual cat
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

# Keep track of assigned colors for each cat
cat_colors = {}

# Plot movement edges with different colors for each cat
for cat_id, group in gdf.groupby('individual-local-identifier'):
    # If the cat doesn't have a color assigned yet, generate one
    if cat_id not in cat_colors:
        cat_colors[cat_id] = generate_random_color()
    
    # Use the assigned color for the cat
    cat_color = cat_colors[cat_id]
    
    # Create a list of coordinates for the cat's path
    path_coordinates = [[row["location-lat"], row["location-long"]] for _, row in group.iterrows()]
    
    # Plot the path (Polyline) for this cat
    folium.PolyLine(path_coordinates, color=cat_color, weight=2, opacity=0.7).add_to(m)
    
    # Plot each point for the cat with the same dot style (uniform for all points)
    for _, row in group.iterrows():
        folium.CircleMarker(
            location=[row["location-lat"], row["location-long"]],
            radius=6,  # Same radius for all dots
            color=cat_color,  # Use the same color for the cat
            fill=True,
            fill_color=cat_color,  # Fill with the same color
            fill_opacity=0.7,  # Same opacity for all dots
            popup=f"Cat ID: {cat_id}, Timestamp: {row['timestamp']}"
        ).add_to(m)
# Save the map to an HTML file (overwriting any existing file with the same name)
m.save("cat_movement_paths_map.html")