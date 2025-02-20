import pandas as pd
import networkx as nx
import sys
sys.path.append("../../Util")
import utils
sys.path.append("../bvns")
import bvns_implementation as bvns
import matplotlib.pyplot as plt


def main():
    input_data = pd.read_csv('SI7_herm.csv', index_col=0) # Read the csv
    input_data.columns = input_data.index # Get column and row names
    input_data = input_data.apply(pd.to_numeric, errors="coerce").fillna(0) # Replace non-numeric entreis with 0
    print(input_data)
    G = nx.from_pandas_adjacency(input_data)
    # Don't actually use these parameters (2 and 5). They're only there so bvns will run quickly
    communities = bvns.bvns(G, 2, stop_criterion=5)
    utils.plot_graph_with_communities(G, communities)

def get_neuronal_connectivity_graph(file_path_from_caller: str = 'SI7_herm.csv') -> nx.Graph:
    input_data = pd.read_csv(file_path_from_caller, index_col=0)  # Read the csv
    input_data.columns = input_data.index # Get column and row names
    input_data = input_data.apply(pd.to_numeric, errors="coerce").fillna(0) # Replace non-numeric entreis with 0

    return nx.from_pandas_adjacency(input_data)


if __name__ == "__main__":
    main()
