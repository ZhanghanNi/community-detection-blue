import fsspec
import geopandas as gpd

# Disable SSL verification
fs = fsspec.filesystem("http", ssl=False)

url = "https://www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"

# Open the file with SSL verification disabled
with fs.open(url) as file:
    gdf = gpd.read_file(file)

# Now gdf contains the geopandas data from the file