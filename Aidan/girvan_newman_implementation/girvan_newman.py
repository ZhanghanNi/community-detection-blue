"""
File for functions used to implement Girvan-Newman algorithm
and the function that implements it itself
"""

import networkx as nx
from typing import List, Set, Tuple

import sys
sys.path.append("../../Util")
import utils

# TODO: add the return type that Yang used and adjust the return statement
def girvan_newman(G: nx.Graph):
    """
    Highest level runner function for Girvan-Newman implementation,
    is called in `girvan_newman_implementation.py`

    Parameters:
    - G: Network X representation of the input graph

    Returns:
    - A list of sets, where each set represents a community in the graph.
    - For now this will be a Network X representation of communities
    """

    optimal_communities: List[Set[int]] = nx.connected_components(G)
    print(f"initial communities: {list(optimal_communities)}")

    max_modularity = utils.modularity(G, optimal_communities)
    print(f"initial modularity: {max_modularity}")

    print(get_max_betweenness_edge(G))

    # while G.edges():
    #     current_communities = nx.connected_components(G)
    #     current_modularity = utils.modularity(G, current_communities)

    #     if current_modularity > current_modularity:
    #         max_modularity = current_modularity
    #         optimal_communities = current_communities

    #     edge_to_remove = get_max_betweenness_edge()


def get_max_betweenness_edge(G: nx.Graph) -> Tuple:
    """
    Find the edge with the highest betweenness centrality

    Parameters:
    - G: Network X representation of a graph

    Returns:
    - Edge with the highest betweenness centrality as a tuple of two nodes
    """

    # TODO: replace Network X implementation with custom implementation
    all_edges_betweenness = nx.edge_betweenness_centrality(G, seed=1234)

    print(f"all_edges_betweenness: {all_edges_betweenness}")

    # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
    return max(all_edges_betweenness, key=all_edges_betweenness.get)
