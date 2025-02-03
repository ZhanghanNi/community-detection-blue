import networkx as nx
import numpy as np
from typing import List, Set, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append("../Util")
import utils

def louvain_method(G: nx.Graph) -> List[Set[int]]:
    """
    Apply the Louvain method for community detection.

    The Louvain method iteratively maximizes modularity to find communities in the graph.

    Parameters:
    - G: The graph.

    Returns:
    - A list of sets, where each set represents a community in the graph.
    """
   # Initialize each node in its own community
    communities = [{node} for node in G.nodes()]
    
    modularity_before = utils.modularity(G, communities)
    #print(f"Initial modularity: {modularity_before}")
    
    # Repeat until modularity stops improving
    while True:
        improvement = False
        for node in G.nodes():
            # Try moving the node to other communities and check modularity gain
            new_modularity, best_community = modularity_gained_by_moving(G, node, communities, modularity_before)
            #print(f"Node {node}: Modularity before: {modularity_before}, after: {new_modularity}")
            
            if new_modularity > modularity_before:
                #print(f"Moving node {node} improves modularity: {new_modularity - modularity_before}")
                communities = best_community
                modularity_before = new_modularity
                improvement = True
        # Add stop point where the user has to press enter to continue
            #input("Press Enter to continue to next iteration...")  # Stops and waits for user input
        if not improvement:
            break
    
    return utils.remove_empty_communities(communities)

def modularity_gained_by_moving(G: nx.Graph, node: int, communities: List[Set[int]], modularity_before: float) -> Tuple[float, List[Set[int]]]:
    """
    Attempt to move a node to another community and compute the modularity gain.
    
    Modularity gain ΔQ is calculated as:
    ΔQ = (1 / 2m) * Σ[i,j] { A_ij - (k_i * k_j) / (2m) } * δ(c_i, c_j)
    
    Parameters:
    - G: The graph.
    - node: The node whose movement is being considered.
    - communities: A list of communities (sets of nodes).
    - modularity_before: The modularity score before moving the node.
    
    Returns:
    - The updated modularity score after the move and the best community for the node.
    """
    best_modularity = modularity_before
    best_communities = communities
    best_gain = 0

    # Remove node from its current community
    communities_without_node = [community - {node} if node in community else community for community in communities]
    
    # Try moving the node to each other community and calculate modularity gain
    for i in range(len(communities)):
        # Create a new set of communities where the node is added to the i-th community
        new_communities = communities_without_node[:i] + [communities[i].union({node})] + communities_without_node[i + 1:]
        new_modularity = utils.modularity(G, new_communities)
        gain = new_modularity - modularity_before
        
        #print(f"Node {node} move to community {i}: New modularity: {new_modularity}, Gain: {gain}")
        
        if gain > best_gain:
            best_gain = gain
            best_modularity = new_modularity
            best_communities = new_communities
    
    return best_modularity, best_communities