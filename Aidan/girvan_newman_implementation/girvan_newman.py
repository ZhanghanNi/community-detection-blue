"""
File for functions used to implement Girvan-Newman algorithm
and the function that implements it itself

Author: Aidan Roessler
"""

import networkx as nx
from typing import List, Set, Tuple, Dict, Deque
from collections import deque
import heapq

import sys

sys.path.append("../../Util")
import utils


def girvan_newman(G: nx.Graph, immutable_G, weighted: bool) -> List[Set[int]]:
    """
    Highest level runner function for Girvan-Newman implementation,
    is called in `girvan_newman_implementation.py`

    Parameters:
    - G: Network X representation of the input graph to be mutated by Girvan-Newman
    - immutable_G: Copy of G in it's original state to use for calculating modularity of different communities
    

    Returns:
    - A list of communities found by the Girvan-Newman algorithm (represented as sets)
    """

    optimal_communities: List[Set[int]] = get_communities(G, weighted)
    max_modularity: float = utils.modularity(immutable_G, optimal_communities)
    #(f"Initial modularity (us): {max_modularity}")

    while G.edges():
        current_communities = get_communities(G, weighted)
        current_modularity = utils.modularity(immutable_G, current_communities)

        if current_modularity > max_modularity:
            max_modularity = current_modularity
            optimal_communities = current_communities

        edge_to_remove = get_max_betweenness_edge(G, weighted)
        G.remove_edge(*edge_to_remove)

    #print(f"Final communities (us): {list(optimal_communities)}")
    #print(f"Final modularity (us): {max_modularity}")

    return optimal_communities

def get_max_betweenness_edge(G: nx.Graph, weighted: bool) -> Tuple:
    """
    Find the edge with the highest betweenness centrality

    Parameters:
    - G: Network X representation of a graph

    Returns:
    - Edge with the highest betweenness centrality as a tuple of two nodes
    """
    # all_edges_betweenness_network_x = nx.edge_betweenness_centrality(
    #     G, seed=1234, weight=None
    # )
    # print(f"all_edges_betweenness (Network X): {all_edges_betweenness_network_x}")

    all_edges_betweenness = weighted_edge_betweenness_centrality(G) if weighted else unweighted_edge_betweenness_centrality(G)
    # print(f"all_edges_betweenness (us): {all_edges_betweenness}")

    # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
    return max(all_edges_betweenness, key=all_edges_betweenness.get)


def get_communities(G: nx.Graph, weighted: bool) -> List[Set[int]]:
    """
    Finds all communities by iterating through all nodes and calling
    get_community_unweighted() with a given node as a parameter if it hasn't
    been visited already

    Parameters:
    - G: Network X representation of a graph

    Returns:
    - All communities of G (all connected components of G)
    """
    visited_nodes: Set[int] = set()
    communities: List[Set[int]] = []

    def get_community_unweighted(root_node) -> Set[int]:
        """
        Find a single community in an unweighted graph based on one of its nodes
        by performing BFS from that `root_node`

        Parameters:
        - root_node: member node of a community

        Returns:
        - Entire community the `root_node` belongs to
        """
        community: Set[int] = set()
        bfs_queue: Deque[int] = deque([root_node])

        community.add(root_node)
        visited_nodes.add(root_node)

        while bfs_queue:
            dequeued_node = bfs_queue.popleft()

            for neighbor in G.neighbors(dequeued_node):
                if neighbor not in visited_nodes:
                    bfs_queue.append(neighbor)
                    community.add(neighbor)
                    visited_nodes.add(neighbor)

        return community

    def get_community_weighted(root_node: int) -> Set[int]:
        """
        Find a single community in a weighted graph based on one of its nodes
        by performing Dijkstra's algorithm from that `root_node`.

        Parameters:
        - root_node: member node of a community

        Returns:
        - Entire community the `root_node` belongs to
        """
        community: Set[int] = set()
        dijkstra_priority_queue: List[Tuple[int, int]] = [(0, root_node)]

        visited_nodes.add(root_node)

        while dijkstra_priority_queue:
            # heapq documentation: https://docs.python.org/3/library/heapq.html
            dequeued_node_distance, dequeued_node = heapq.heappop(dijkstra_priority_queue)

            if dequeued_node in community:
                continue

            community.add(dequeued_node)

            for neighbor in G.neighbors(dequeued_node):
                if neighbor not in visited_nodes:
                    edge_weight = G[dequeued_node][neighbor].get('weight', 1)
                    heapq.heappush(dijkstra_priority_queue, (dequeued_node_distance + edge_weight, neighbor))
                    visited_nodes.add(neighbor)

        return community

    # Traverse through all communities
    for node in G.nodes():
        if node not in visited_nodes:
            # Python Ternary operator: https://www.geeksforgeeks.org/ternary-operator-in-python/
            communities.append(
                get_community_weighted(node)
                if weighted
                else get_community_unweighted(node)
            )

    return communities


def unweighted_edge_betweenness_centrality(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness centrality of every edge in the given
    unweighted graph inspired by the pseudocode algorithm to calculate edge betweenness for nodes from:
    A Faster Algorithm for Betweenness Centrality by Ulrik Brandes https://doi.org/10.1080/0022250X.2001.9990249

    Parameters:
    - G: Network X representation of a graph

    Returns:
    - A dictionary mapping each edge to its betweenness centrality
    """
    # List comprehension syntax inspired by: https://www.w3schools.com/python/python_lists_comprehension.asp
    # Deque syntax inspired by: https://www.geeksforgeeks.org/python-perform-append-at-beginning-of-list/

    # Populate an initial dictionary with 0 values for all edges
    betweenness_centrality: Dict[Tuple[int, int], float] = {
        edge: 0.0 for edge in G.edges()
    }

    # s = source node
    for s in G.nodes():
        # Setup all we need to calculate dependency
        processing_stack: Deque[int] = deque([])

        predecessors: Dict[int, List[int]] = {node: [] for node in G.nodes()}

        num_shortest_paths_from_s: Dict[int, int] = {node: 0 for node in G.nodes()}
        num_shortest_paths_from_s[s] = 1

        distance_from_s: Dict[int, int] = {node: -1 for node in G.nodes()}
        distance_from_s[s] = 0

        # Perform BFS to populate all variables initialized above
        bfs_queue: Deque[int] = deque([])
        bfs_queue.append(s)

        while bfs_queue:
            dequeued_node = bfs_queue.popleft()
            processing_stack.appendleft(dequeued_node)

            for neighbor in G.neighbors(dequeued_node):
                # Neighbor node found for the first time?
                if distance_from_s[neighbor] < 0:
                    bfs_queue.append(neighbor)
                    distance_from_s[neighbor] = distance_from_s[dequeued_node] + 1

                # Is there a new shortest path from s to neighbor through dequeued_node?
                if distance_from_s[neighbor] == distance_from_s[dequeued_node] + 1:
                    num_shortest_paths_from_s[neighbor] += num_shortest_paths_from_s[
                        dequeued_node
                    ]
                    predecessors[neighbor].append(dequeued_node)

        # Use the variables we initialized with BFS to calculate each node's
        # dependency and then use that dependency to calculate the betweenness centrality
        # for the edges attached to it
        dependency: Dict[int, float] = {node: 0 for node in G.nodes()}

        while processing_stack:
            popped_node = processing_stack.popleft()

            for predecessor in predecessors[popped_node]:
                proportion_of_shortest_paths = (
                    num_shortest_paths_from_s[predecessor]
                    / num_shortest_paths_from_s[popped_node]
                )
                dependency[predecessor] += proportion_of_shortest_paths * (
                    1 + dependency[popped_node]
                )

                edge_to_update: Tuple[int, int] = tuple(
                    sorted(
                        (predecessor, popped_node)
                    )  # Ensures (small, large) order that Network X uses
                )
                betweenness_centrality[
                    edge_to_update
                ] += proportion_of_shortest_paths * (1 + dependency[popped_node])

    return betweenness_centrality


def weighted_edge_betweenness_centrality(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness centrality of every edge in the given
    weighted graph inspired by the pseudocode algorithm to calculate edge betweenness for nodes from:
    A Faster Algorithm for Betweenness Centrality by Ulrik Brandes https://doi.org/10.1080/0022250X.2001.9990249

    Parameters:
    - G: Network X representation of a graph

    Returns:
    - A dictionary mapping each edge to its betweenness centrality
    """
    # List comprehension syntax inspired by: https://www.w3schools.com/python/python_lists_comprehension.asp
    # Deque syntax inspired by: https://www.geeksforgeeks.org/python-perform-append-at-beginning-of-list/

    # Populate an initial dictionary with 0 values for all edges
    betweenness_centrality: Dict[Tuple[int, int], float] = {
        edge: 0.0 for edge in G.edges()
    }

    # s = source node
    for s in G.nodes():
        # Setup all we need to calculate dependency
        processing_stack: Deque[int] = deque([])

        predecessors: Dict[int, List[int]] = {node: [] for node in G.nodes()}

        num_shortest_paths_from_s: Dict[int, int] = {node: 0 for node in G.nodes()}
        num_shortest_paths_from_s[s] = 1

        distance_from_s: Dict[int, int] = {node: -1 for node in G.nodes()}
        distance_from_s[s] = 0

        # Perform BFS to populate all variables initialized above
        dijkstra_priority_queue: List[Tuple[int, int]] = [(0, s)]

        while dijkstra_priority_queue:
            dequeued_node_distance, dequeued_node = heapq.heappop(
                dijkstra_priority_queue
            )
            processing_stack.appendleft(dequeued_node)

            for neighbor in G.neighbors(dequeued_node):
                edge_weight = G[dequeued_node][neighbor].get('weight', 1)
                
                # Neighbor node found for the first time?
                if distance_from_s[neighbor] < 0:
                    heapq.heappush(
                        dijkstra_priority_queue,
                        (dequeued_node_distance + edge_weight, neighbor),
                    )
                    distance_from_s[neighbor] = distance_from_s[dequeued_node] + edge_weight

                # Is there a new shortest path from s to neighbor through dequeued_node?
                if distance_from_s[neighbor] == distance_from_s[dequeued_node] + edge_weight:
                    num_shortest_paths_from_s[neighbor] += num_shortest_paths_from_s[
                        dequeued_node
                    ]
                    predecessors[neighbor].append(dequeued_node)

        # Use the variables we initialized with BFS to calculate each node's
        # dependency and then use that dependency to calculate the betweenness centrality
        # for the edges attached to it
        dependency: Dict[int, float] = {node: 0 for node in G.nodes()}

        while processing_stack:
            popped_node = processing_stack.popleft()

            for predecessor in predecessors[popped_node]:
                proportion_of_shortest_paths = (
                    num_shortest_paths_from_s[predecessor]
                    / num_shortest_paths_from_s[popped_node]
                )
                dependency[predecessor] += proportion_of_shortest_paths * (
                    1 + dependency[popped_node]
                )

                edge_to_update: Tuple[int, int] = tuple(
                    sorted(
                        (predecessor, popped_node)
                    )  # Ensures (small, large) order that Network X uses
                )
                betweenness_centrality[
                    edge_to_update
                ] += proportion_of_shortest_paths * (1 + dependency[popped_node])

    return betweenness_centrality
