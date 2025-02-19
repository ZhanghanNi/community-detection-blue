import networkx as nx
import numpy as np
from typing import List, Set, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append("../../Util")
import utils

def louvain_method(G: nx.Graph) -> List[Set[int]]:
    """
    Apply the Louvain method for community detection with iterative merging.

    Parameters:
    - G: The graph.

    Returns:
    - A list of sets, where each set represents a community in the graph.
    """
    communities = [{node} for node in G.nodes()]
    modularity_before = utils.modularity(G, communities)
    
    while True:
        improvement = False
        for node in G.nodes():
            new_modularity, best_community = modularity_gained_by_moving(G, node, communities, modularity_before)
            
            if new_modularity > modularity_before:
                communities = best_community
                modularity_before = new_modularity
                improvement = True
        
        if not improvement:
            break
        
        # Merge communities into a new graph representation
        G = merge_communities(G, communities)
    
    return communities

def modularity_gained_by_moving(G: nx.Graph, node: int, communities: List[Set[int]], modularity_before: float) -> Tuple[float, List[Set[int]]]:
    """
    Attempt to move a node to another community and compute the modularity gain.
    """
    best_modularity = modularity_before
    best_communities = communities
    best_gain = 0

    communities_without_node = [community - {node} if node in community else community for community in communities]
    
    for i in range(len(communities)):
        new_communities = communities_without_node[:i] + [communities[i].union({node})] + communities_without_node[i + 1:]
        new_modularity = utils.modularity(G, new_communities)
        gain = new_modularity - modularity_before
        
        if gain > best_gain:
            best_gain = gain
            best_modularity = new_modularity
            best_communities = new_communities
    
    return best_modularity, best_communities

def merge_communities(G: nx.Graph, communities: List[Set[int]]) -> nx.Graph:
    """
    Merge communities into a new graph where each node represents a community.

    Parameters:
    - G: The original graph.
    - communities: List of sets where each set is a community.

    Returns:
    - A new graph where each node is a set representing a merged community.
    """
    new_G = nx.Graph()
    community_map = {node: i for i, community in enumerate(communities) for node in community}
    
    for i, community in enumerate(communities):
        new_G.add_node(frozenset(community))
    
    for u, v in G.edges():
        comm_u = community_map[u]
        comm_v = community_map[v]
        
        if comm_u != comm_v:
            u_set = frozenset(communities[comm_u])
            v_set = frozenset(communities[comm_v])
            
            if new_G.has_edge(u_set, v_set):
                new_G[u_set][v_set]['weight'] += G[u][v].get('weight', 1)
            else:
                new_G.add_edge(u_set, v_set, weight=G[u][v].get('weight', 1))
    
    return new_G