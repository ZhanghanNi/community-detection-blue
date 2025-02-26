import networkx as nx
import numpy as np
from typing import List, Set, Tuple
import matplotlib.pyplot as plt
import sys
sys.path.append("../Util")
import utils

def louvain_method_sub(G: nx.Graph) -> List[Set[int]]:
    """
    Apply the Louvain method for community detection considering adjacency between communities.

    Parameters:
    - G: The graph.

    Returns:
    - A list of sets, where each set represents a community in the graph.
    """
    # Initialize each node in its own community
    communities = [{node} for node in G.nodes()]
    
    modularity_before = utils.modularity(G, communities)
    
    # Get adjacency matrix for later use
    adj_matrix = nx.to_numpy_array(G)
    
    # Repeat until modularity stops improving
    while True:
        improvement = False
        for node in G.nodes():
            # Get adjacent communities using adjacency matrix
            adjacent_communities = find_adjacent_communities(G, node, communities, adj_matrix)
            
            # Try moving the node to adjacent communities and check modularity gain
            new_modularity, best_community = modularity_gained_by_moving(
                G, node, communities, modularity_before, adjacent_communities
            )
            
            if new_modularity > modularity_before:
                communities = best_community
                modularity_before = new_modularity
                improvement = True
        if not improvement:
            break
    
    return utils.remove_empty_communities(communities)

def find_adjacent_communities(
    G: nx.Graph, node: int, communities: List[Set[int]], adj_matrix: np.ndarray
) -> List[int]:
    """
    Find the indices of communities that are adjacent to the given node using the adjacency matrix.

    Parameters:
    - G: The graph.
    - node: The node whose adjacent communities are being found.
    - communities: A list of communities (sets of nodes).
    - adj_matrix: The adjacency matrix of the graph.

    Returns:
    - A list of indices of adjacent communities.
    """
    node_idx = list(G.nodes()).index(node)
    adjacent_communities = set()

    # Check the adjacency of the node to other nodes in the graph
    for neighbor_idx, is_adjacent in enumerate(adj_matrix[node_idx]):
        if is_adjacent > 0:  # There's an edge
            neighbor_node = list(G.nodes())[neighbor_idx]
            #print(f"{node_idx}: {str(adjacent_communities)}")
            for i, community in enumerate(communities):
                if neighbor_node in community:
                    adjacent_communities.add(i)

    #print(f"{node_idx}: {str(adjacent_communities)}")
    return list(adjacent_communities)

def modularity_gained_by_moving(
    G: nx.Graph, node: int, communities: List[Set[int]], modularity_before: float, adjacent_communities: List[int]
) -> Tuple[float, List[Set[int]]]:
    """
    Attempt to move a node to another adjacent community and compute the modularity gain.

    Parameters:
    - G: The graph.
    - node: The node whose movement is being considered.
    - communities: A list of communities (sets of nodes).
    - modularity_before: The modularity score before moving the node.
    - adjacent_communities: A list of indices of adjacent communities.

    Returns:
    - The updated modularity score after the move and the best community for the node.
    """
    best_modularity = modularity_before
    best_communities = communities
    best_gain = 0

    # Remove node from its current community
    communities_without_node = [community - {node} if node in community else community for community in communities]
    
    # Try moving the node to each adjacent community and calculate modularity gain
    for i in adjacent_communities:
        # Create a new set of communities where the node is added to the i-th adjacent community
        new_communities = communities_without_node[:i] + [communities[i].union({node})] + communities_without_node[i + 1:]
        new_modularity = utils.modularity(G, new_communities)
        gain = new_modularity - modularity_before
        
        if gain > best_gain:
            best_gain = gain
            best_modularity = new_modularity
            best_communities = new_communities
    
    return best_modularity, best_communities

def update_communities(communities: List[Set[int]], new_communities: List[Set[int]]) -> List[Set[int]]:
    """
    Update the current community structure by merging communities based on the result of the Louvain method.

    Parameters:
    - communities: The current community structure (list of sets of nodes).
    - new_communities: The new community structure (list of sets of indices referring to communities in 'communities').

    Returns:
    - A list of updated communities after merging the specified indices.
    """
    merged_communities = []
    merged_indices = set()

    for indices in new_communities:
        merged_community = set()
        for idx in indices:
            if idx < len(communities):  # Ensure index is valid
                merged_community.update(communities[idx])
        merged_communities.append(merged_community)
        merged_indices.update(indices)

    # Add any communities that were not merged
    for i, community in enumerate(communities):
        if i not in merged_indices:
            merged_communities.append(community)

    return merged_communities

def louvain_method(G: nx.Graph, max_iterations: int = 0) -> List[Set[int]]:
    """
    Run the Louvain-based community detection algorithm on a given graph for a fixed number of iterations.

    Parameters:
    - G: The input graph.
    - max_iterations: The number of iterations to run the algorithm.

    Returns:
    - The final community structure.
    """
    communities = louvain_method_sub(G)
    #print(f"Initial communities: {communities}")

    #previous_communities = None

    for iteration in range(0, max_iterations):
        #previous_communities = communities

        # Merge communities in the graph
        merged_G = utils.merge_communities(G, communities)
        #print(f"Graph after merging communities (iteration {iteration}): {merged_G.nodes()}")

        # Apply Louvain method to the merged graph
        new_communities = louvain_method_sub(merged_G)
        #print(f"New communities after iteration {iteration}: {new_communities}")

        # Update the community structure
        communities = update_communities(communities, new_communities)

        # Always print whether convergence happened, but don't break early
        #if communities == previous_communities:
            #print(f"Communities have converged after {iteration} iterations.")

    return communities

