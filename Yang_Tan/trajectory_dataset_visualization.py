import folium
import pandas as pd
import geopandas as gpd
import random

# Load the dataset (ensure the file path is correct)
file_path = "your_dataset.csv"  # Adjust this path to your dataset location
df = pd.read_csv(file_path)

# Convert the timestamp column to datetime (ensure it's named appropriately in your dataset)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Select relevant columns (customize this based on your dataset's column names)
id_column = "trajectory-id"  # Change this to the appropriate column for unique IDs
lat_column = "location-lat"  # Adjust based on your dataset
lon_column = "location-long"  # Adjust based on your dataset
timestamp_column = "timestamp"  # Adjust based on your dataset

# Filter the DataFrame (if needed, adjust the time range or skip this step)
start_time = pd.to_datetime("2015-06-21 23:59:00")
end_time = pd.to_datetime("2015-12-21 23:59:00")
df_filtered = df[(df[timestamp_column] >= start_time) & (df[timestamp_column] <= end_time)]

# Extract relevant columns
df_extracted = df_filtered[[id_column, lat_column, lon_column, timestamp_column]].copy()

# Check how many unique trajectories (id_column) are in the filtered data
unique_trajectories = df_extracted[id_column].nunique()
print(f"Number of unique trajectories in the selected time range: {unique_trajectories}")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df_extracted,
    geometry=gpd.points_from_xy(df_extracted[lon_column], df_extracted[lat_column]),
    crs="EPSG:4326"  # WGS 84
)

# Sort by trajectory ID and timestamp
gdf = gdf.sort_values(by=[id_column, timestamp_column])

# Create the folium map (you can adjust the center and zoom level based on your dataset)
m = folium.Map(location=[35.7796, -78.6382], zoom_start=12)

# Generate a consistent random color for each trajectory
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

# Keep track of assigned colors for each trajectory
trajectory_colors = {}

# Plot movement edges with different colors for each trajectory
for trajectory_id, group in gdf.groupby(id_column):
    # If the trajectory doesn't have a color assigned yet, generate one
    if trajectory_id not in trajectory_colors:
        trajectory_colors[trajectory_id] = generate_random_color()
    
    # Use the assigned color for the trajectory
    trajectory_color = trajectory_colors[trajectory_id]
    
    # Create a list of coordinates for the trajectory's path
    path_coordinates = [[row[lat_column], row[lon_column]] for _, row in group.iterrows()]
    
    # Plot the path (Polyline) for this trajectory
    folium.PolyLine(path_coordinates, color=trajectory_color, weight=2, opacity=0.7).add_to(m)
    
    # Plot each point for the trajectory with the same dot style (uniform for all points)
    for _, row in group.iterrows():
        folium.CircleMarker(
            location=[row[lat_column], row[lon_column]],
            radius=6,  # Same radius for all dots
            color=trajectory_color,  # Use the same color for the trajectory
            fill=True,
            fill_color=trajectory_color,  # Fill with the same color
            fill_opacity=0.7,  # Same opacity for all dots
            popup=f"{id_column}: {trajectory_id}, Timestamp: {row[timestamp_column]}"
        ).add_to(m)

# Save the map to an HTML file (overwriting any existing file with the same name)
m.save("trajectory_movement_paths_map.html")