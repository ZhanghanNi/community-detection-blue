# README for Yang's Folder

Only follow the instructions in this README if you are trying to run the `gps_visualization.py` script which renders a visualization of our old GPS data. If you are trying to run the `louvain_method.py` script, follow the instructions in the README in the root directory of the repository.

## GPS Visualization (`gps_visualization.py`)

### Prerequisites

1. **Install `networkx` and `python-louvain`:**
   ```bash
   pip install networkx
   ```

2. **Install `geopandas`:**
   - The recommended method is using `conda`, especially for handling complex binary dependencies:
     ```bash
     conda install -c conda-forge geopandas
     ```
   - Alternatively, you can use pip for simpler installations:
     ```bash
     pip install geopandas
     ```
   - For more details on installing Geopandas, check their [official guide](https://geopandas.org/en/stable/getting_started/install.html).

3. **Install `geodatasets`:**
   - Use `conda` to install the package:
     ```bash
     conda install -c conda-forge geodatasets
     ```
   - or `pip`
     ```bash
     pip install geodatasets
     ```

4. **Install `fiona`:**
   - You can install it with pip:
     ```bash
     pip install fiona
     ```
install pyproj
conda config --prepend channels conda-forge
conda config --set channel_priority strict
conda create -n pyproj_env pyproj
conda activate pyproj_env

pip install --upgrade botocore boto3

### Additional Resources
- Geopandas documentation: [Geopandas Official Docs](https://geopandas.org/en/stable/index.html).
- If you’re new to `conda`, you can find the installation guide for Miniconda [here](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).

conda install contextily --channel conda-forge


---
test

pip3 install fsspec gcsfs


