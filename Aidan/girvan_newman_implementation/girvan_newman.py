"""
File for functions used to implement Girvan-Newman algorithm
and the function that implements it itself

Author: Aidan Roessler
"""

import networkx as nx
from typing import List, Set, Tuple

import sys
sys.path.append("../../Util")
import utils

# TODO: add the return type that Yang used and adjust the return statement
def girvan_newman(G: nx.Graph) -> List[Set[int]]:
    """
    Highest level runner function for Girvan-Newman implementation,
    is called in `girvan_newman_implementation.py`

    Parameters:
    - G: Network X representation of the input graph

    Returns:
    - A list of communities found by the Girvan-Newman algorithm (represented as sets)
    """

    # optimal_communities: List[Set[int]] = nx.connected_components(G)
    optimal_communities = get_communities(G)
    print(f"initial communities: {list(optimal_communities)}")

    max_modularity = utils.modularity(G, optimal_communities)
    print(f"initial modularity: {max_modularity}")

    print(get_max_betweenness_edge(G))

    while G.edges():
        current_communities = get_communities(G)
        current_modularity = utils.modularity(G, current_communities)

        if current_modularity > max_modularity:
            max_modularity = current_modularity
            optimal_communities = current_communities

        edge_to_remove = get_max_betweenness_edge(G)
        G.remove_edge(*edge_to_remove)

    print(f"Final optimal_communities: {list(optimal_communities)}")
    print(f"Final modularity: {max_modularity}")
    
    return optimal_communities


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


def get_communities(G: nx.Graph) -> List[Set[int]]:
    """
    Finds all communities by iterating through all nodes and calling 
    get_community() with a given node as a parameter if it hasn't 
    been visited already
    
    Parameters:
    - G: Network X representation of a graph
    
    Returns:
    - All communities of G (all connected components of G)
    """
    visited_nodes: Set[int] = set()
    communities: List[Set[int]] = []

    def get_community(root_node) -> Set[int]:
        """
        Find a single community based on one of its nodes
        by performing BFS from that `root_node`
        
        Parameters:
        - root_node: member node of a community
        
        Returns:
        - Entire community the `root_node` belongs to
        """
        community = set()
        neighbors_queue = [root_node]

        community.add(root_node)
        visited_nodes.add(root_node)

        while neighbors_queue:
            popped_node = neighbors_queue.pop(0)

            for neighbor in G.neighbors(popped_node):
                if neighbor not in visited_nodes:
                    neighbors_queue.append(neighbor)
                    community.add(neighbor)
                    visited_nodes.add(neighbor)

        print("community found:")
        print(community)
        return community

    # Traverse through all communities
    for node in G.nodes():
        if node not in visited_nodes:
            communities.append(get_community(node))
            
    return communities
