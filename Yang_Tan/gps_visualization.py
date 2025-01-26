import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geodatasets import get_path
from shapely.geometry import LineString

#Source Code
df = pd.DataFrame(
    {
        "City": ["Buenos Aires", "Brasilia", "Santiago", "Bogota", "Caracas"],
        "Country": ["Argentina", "Brazil", "Chile", "Colombia", "Venezuela"],
        "Latitude": [-34.58, -15.78, -33.45, 4.60, 10.48],
        "Longitude": [-58.66, -47.91, -70.66, -74.08, -66.86],
    }
)

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)

print(gdf.head())

world = gpd.read_file(get_path("naturalearth.land"))

# Create edges (lines) between each point
lines = []
for i in range(len(gdf)):
    for j in range(i + 1, len(gdf)):
        point1 = gdf.geometry[i]
        point2 = gdf.geometry[j]
        lines.append(LineString([point1, point2]))

# Create GeoDataFrame for lines
edges_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

# We restrict to South America.
ax = world.clip([-90, -55, -25, 15]).plot(color="white", edgecolor="black")

# Plot points and edges
gdf.plot(ax=ax, color="red", label="Cities")
edges_gdf.plot(ax=ax, color="blue", linewidth=0.5, label="Edges")

plt.legend()
plt.show()