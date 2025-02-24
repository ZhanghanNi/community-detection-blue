import pandas as pd
import networkx as nx
from cdlib import algorithms as al
import sys
sys.path.append("../../Util")
import utils
sys.path.append("../bvns")
import bvns_implementation as bvns
import matplotlib.pyplot as plt
from collections import Counter

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
            
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

# Assuming G is your directed graph
num_self_loops = nx.number_of_selfloops(G)
print("Number of self-loops:", num_self_loops)    

# Assuming G is your directed graph
edges = G.edges()

# Count occurrences of each edge
edge_counts = Counter(edges)

def count_duplicate_edges(G):
    count = 0
    for u, v in G.edges():
        if G.has_edge(v, u):  # Check if the reverse edge exists
            count += 1
    return count // 2  # Each duplicate is counted twice, so divide by 2

# duplicate_edges_count = count_duplicate_edges(G)
# print(f"Number of duplicate edges: {duplicate_edges_count}")

# # Print the number of duplicate edges
# print(f"Number of duplicate edges: {duplicate_edges_count}")

# self_loops_directed = sum(1 for node in G.nodes if G.has_edge(node, node))
# print(f"Self-loops in directed graph: {self_loops_directed}")

# G = G.to_undirected()

# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()

# print(f"Number of nodes: {num_nodes}")
# print(f"Number of edges: {num_edges}")

# self_loops_undirected = sum(1 for node in G.nodes if G.has_edge(node, node))
# print(f"Self-loops in undirected graph: {self_loops_undirected}")

def check_graph_properties(graph, name="Graph"):
    print(f"=== {name} ===")
    
    # Number of nodes and edges
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

    # Count self-loops
    num_self_loops = sum(1 for node in graph.nodes if graph.has_edge(node, node))
    print(f"Number of self-loops: {num_self_loops}")

    # Count duplicate edges (edges going both directions)
    duplicate_edges = sum(1 for u, v in graph.edges if graph.has_edge(v, u) and u != v) // 2
    print(f"Number of duplicate edges (bidirectional edges counted once): {duplicate_edges}")

    print("=" * 30 + "\n")

# Check properties of the directed graph
check_graph_properties(G, "Directed Graph")

# Convert to undirected
G_undirected = G.to_undirected()

# Check properties of the undirected graph
check_graph_properties(G_undirected, "Undirected Graph")
