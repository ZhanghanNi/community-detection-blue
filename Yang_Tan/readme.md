# README

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

---
test

pip3 install fsspec gcsfs



## Louvain Method (`louvain_method.py`)

### Prerequisites

1. **Install `networkx` and `python-louvain`:**
   ```bash
   pip install networkx
   ```

--- 

### Notes
- This README provides setup instructions for running the `gps_visualization.py` and `louvain_method.py` scripts.
- Ensure all dependencies are installed before running the scripts to avoid errors.
