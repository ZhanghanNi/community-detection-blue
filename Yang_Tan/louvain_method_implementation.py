import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import louvain_method as lm
from typing import List, Set, Tuple

import sys
sys.path.append("../Util")
import utils


def main():
    # # Create a sample graph
    # G = nx.Graph()
    # # Add nodes with attributes
    # G.add_node(1, label="A")
    # G.add_node(2, label="B")
    # G.add_node(3, label="C")
    # G.add_node(4, label="D")
    # G.add_node(5, label="E")
    # G.add_node(6, label="F")
    # G.add_node(7, label="G")
    # G.add_node(8, label="H")

    # # Add weighted edges
    # G.add_edges_from([
    #     (1, 2, {'weight': 1}),
    #     (1, 3, {'weight': 2}),
    #     (2, 3, {'weight': 1}),
    #     (3, 4, {'weight': 1}),
    #     (4, 5, {'weight': 2}),
    #     (5, 6, {'weight': 1}),
    #     (6, 7, {'weight': 2}),
    #     (5, 7, {'weight': 1}),
    #     (6, 8, {'weight': 2}),
    #     (7, 8, {'weight': 1}),
    # ])
    G = nx.karate_club_graph()
    # # Assign weights to edges
    # for u, v in G.edges:
    # G[u][v]["weight"] = 1.0

    print("Edge Weights:")
    for u, v, data in G.edges(data=True):
        print(f"({u}, {v}) - weight: {data['weight']}")
    # Apply Louvain method
    communities = lm.louvain_method(G)

    # Print the detected communities
    print("\nDetected communities:")
    for i, community in enumerate(communities):
        print(f"Community {i + 1}: {community}")

    # Plot the graph with communities
    utils.plot_graph_with_communities(G, communities)
    merged_G = lm.merge_communities(
        G, communities
    )
    print("\nAfter merging:")
    utils.plot_graph_with_communities(
        merged_G,
        communities=[{node} for node in merged_G.nodes()],
        label_edges=True,
        title="Graph with Communities Detected by Louvain Method",
    )


if __name__ == "__main__":
    main()
