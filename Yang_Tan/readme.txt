# For gps_visualization.py
geopandas information: https://geopandas.org/en/stable/index.html
1. You need to install geopandas to run this: https://geopandas.org/en/stable/getting_started/install.html
   You can do this by 
   $pip install geopandas
   However, pip is good for some simple installation tasks, such as pure-python installs. If complex binary deps are 
   getting you down, a more sophisticated tool would be a better match for your needs.
    Create / activate an empty conda environment, and then follow the install instructions:
    $ conda install -c conda-forge geopandas
    For code installation, check here: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
2. Moreover, you need 'geodatasets' package. Please instaill it with:
    $conda install conda-forge::geodatasets

3. And apparentlly you need fiona:
    $ pip install fiona
