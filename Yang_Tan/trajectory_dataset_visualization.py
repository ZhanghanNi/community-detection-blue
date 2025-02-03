import folium
import pandas as pd
import geopandas as gpd
import os
import glob
import random
from datetime import datetime, timedelta

output_folder = "html"
output_file = "user_trajectories_map.html"
output_path = os.path.join(output_folder, output_file)


base_dir = "Data"
all_data = []
date = "20081024"

# List to store user IDs that have matching files
valid_user_ids = []

# Loop through each user folder
for user_id in range(0, 182):  # User folders range from 000 to 182
    folder_path = os.path.join(base_dir, f"{user_id:03d}", "Trajectory")  # Navigate to the Trajectory folder

    # Find all .plt files matching the YYYYMMDDxxxx.plt pattern
    plt_files = glob.glob(os.path.join(folder_path, f"{date}*.plt"))  # Only files starting with '2008'
    
    if plt_files:  # If matching files are found
        valid_user_ids.append(user_id)  # Store the valid user_id

    for file_path in plt_files:
        with open(file_path, "r") as file:
            lines = file.readlines()[6:]  # Skip the first 6 lines

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue  # Skip malformed lines
            
            lat = float(parts[0])
            lon = float(parts[1])
            date_str = parts[5]
            time_str = parts[6]
            timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            
            all_data.append([user_id, lat, lon, timestamp])


# Convert to DataFrame
df = pd.DataFrame(all_data, columns=["user_id", "latitude", "longitude", "timestamp"])

# Sort by user_id and timestamp
df = df.sort_values(by=["user_id", "timestamp"])

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
    crs="EPSG:4326"
)

# Create a folium map (adjust center and zoom based on dataset)
m = folium.Map(location=[39.90, 116.38], zoom_start=12)  # Beijing (example)

# Generate a unique color for each user
def generate_random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

user_colors = {}

# Plot movement paths and points
for user_id, group in gdf.groupby("user_id"):
    if user_id not in user_colors:
        user_colors[user_id] = generate_random_color()
    
    user_color = user_colors[user_id]
    path_coordinates = [[row["latitude"], row["longitude"]] for _, row in group.iterrows()]
    
    # Plot the path
    folium.PolyLine(path_coordinates, color=user_color, weight=2, opacity=0.7).add_to(m)
    
    # Plot points
    for _, row in group.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=user_color,
            fill=True,
            fill_color=user_color,
            fill_opacity=0.7,
            popup=f"User ID: {user_id}, Timestamp: {row['timestamp']}"
        ).add_to(m)


# Save the map inside the specified folder
m.save(output_path)


# Print the total count and list of user IDs with matching files
print(f"\nTotal Users with matching .plt files: {len(valid_user_ids)}")
print("List of User IDs:", valid_user_ids)