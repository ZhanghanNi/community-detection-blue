import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import louvain_method as lm
from typing import List, Set, Tuple

import sys
sys.path.append("../Util")
import utils
def assign_uniform_weights(G: nx.Graph, weight: float = 1.0) -> None:
    """
    If the graph is unweighted, assign a uniform weight to all edges.
    """
    if not nx.is_weighted(G, weight='weight'):
        for u, v in G.edges():
            G[u][v]['weight'] = weight




def main(G: nx.Graph, dataset_name: str = "Graph")-> List[Set[int]]:
    #assign_uniform_weights(G, weight=1.0)
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
    # # Assign weights to edges
    # for u, v in G.edges:
    # G[u][v]["weight"] = 1.0

    #print("Edge Weights:")
   # for u, v, data in G.edges(data=True):
     #   print(f"({u}, {v}) - weight: {data['weight']}")
    # Apply Louvain method
    communities = lm.louvain_method(G)
    #print(communities)

    # Print the detected communities
    # print("\nDetected communities:")
    # for i, community in enumerate(communities):
    #   print(f"Community {i + 1}: {community}")

    print(communities)

    # Plot the graph with communities
    utils.plot_graph_with_communities(
        G,
        communities,
        title=f"{dataset_name} with Communities Detected by our Implementation of the Louvain Method",
    )
    merged_G = utils.merge_communities(
        G,
        communities,
    )
    #print("\nAfter merging:")
    utils.plot_graph_with_communities(
        merged_G,
        communities=[{node} for node in merged_G.nodes()],
        label_edges=True,
        title=f"{dataset_name} with Communities Detected by our Implementation of the Louvain Method",
    )
    return communities


if __name__ == "__main__":
    main()
