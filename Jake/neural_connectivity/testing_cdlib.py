import pandas as pd
import networkx as nx
from cdlib import algorithms as al
import sys
sys.path.append("../../Util")
import utils
sys.path.append("../bvns")
import bvns_implementation as bvns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "combined.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info(), df.head()

# Extract neuron names
neurons = df.columns[1:]  # Exclude the first column which contains row labels

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for index, row in df.iterrows():
    pre_synaptic = row.iloc[0]  # The first column contains the pre-synaptic neuron name
    for post_synaptic, weight in row.iloc[1:].items():  # Remaining columns are post-synaptic neurons
        if pd.notna(weight):  # Only add edges for non-NaN values
            G.add_edge(pre_synaptic, post_synaptic, weight=weight)

# Basic graph stats
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print("Edges: ", num_edges)
print("Nodes: ", num_nodes)

best_modularity = -1
best_communities = []
for i in range(10000):
    communities = al.rb_pots(G)
    mod = nx.community.modularity(G, communities.communities)
    if mod > best_modularity:
        best_communities = communities
        best_modularity = mod
    if i % 1000 == 0:
        print(i)

# print(communities.communities)

for i, comm in enumerate(best_communities.communities):
    print(f"Community {i+1}: {comm}")
    
print("Modularity: ", nx.community.modularity(G, best_communities.communities))