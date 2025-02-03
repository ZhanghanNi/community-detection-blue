"""
File for storing shared utility functions used in different implementations
(e.g. calculating modularity with modularity())

Authors: Yang Tan, Aidan Roessler
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Set, Tuple

def modularity(G: nx.Graph, communities: List[Set[int]]) -> float:
    """
    Calculate the modularity Q of a partitioning of the graph.

    Modularity is calculated as:
    Q = (1 / 2m) * Σ[i,j] { A_ij - (k_i * k_j) / (2m) } * δ(c_i, c_j)

    where:
    - A_ij is the weight of the edge between nodes i and j,
    - k_i and k_j are the degrees of nodes i and j,
    - m is the total weight of the edges in the graph,
    - δ(c_i, c_j) is 1 if nodes i and j belong to the same community, and 0 otherwise.

    Parameters:
    - G: The graph.
    - communities: A list of sets where each set is a community of nodes.

    Returns:
    - The modularity score of the partitioning.
    
    Author: Yang Tan
    """
    m = G.size(weight="weight")  # Total weight of the edges in the graph
    Q = 0.0
    for community in communities:
        internal_weight = sum(
            [
                G[u][v].get("weight", 1)
                for u in community
                for v in community
                if G.has_edge(u, v)
            ]
        )

        degree_sum = sum([G.degree(node, weight="weight") for node in community])
        # print(f"Community: {community}, internal_weight: {internal_weight}, degree_sum: {degree_sum}")

        Q += internal_weight / (2 * m) - (degree_sum / (2 * m)) ** 2

    return Q


def plot_graph(G: nx.Graph, label_edges: bool = False, title: str = "Graph"):
    """
    Plot the graph with nodes colored according to their communities.

    Parameters:
    - G: Network X representation of a graph

    Authors: Yang Tan, Aidan Roessler
    """

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=1234)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, cmap=plt.cm.rainbow, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Optionally label edges with their weights
    if label_edges:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Show the plot
    plt.title(title)
    plt.show()


def plot_graph_with_communities(
    G: nx.Graph,
    communities: List[Set[int]],
    label_edges: bool = False,
    title="Graph with Communities",
):
    """
    Plot the graph with nodes colored according to their communities.

    Parameters:
    - G: The graph.
    - communities: The list of communities.

    Authors: Yang Tan, Aidan Roessler
    """
    # Assign a color to each community
    community_colors = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_colors[node] = idx

    # Create a list of node colors based on the community
    node_colors = [community_colors[node] for node in G.nodes()]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=1234)  # positions for all nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=plt.cm.rainbow, node_size=500
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Optionally label edges with their weights
    if label_edges:
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Create legend for communities
    community_patches = []
    for idx, community in enumerate(communities):
        # Each community will have a unique color
        patch = mpatches.Patch(
            color=plt.cm.rainbow(idx / len(communities)), label=f"Community {idx + 1}"
        )
        community_patches.append(patch)

    # Adjust the position of the legend to avoid covering the graph
    plt.legend(
        handles=community_patches,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Communities",
    )

    # Adjust layout to make space for the legend
    plt.tight_layout()

    # Show the plot
    plt.title(title)
    plt.show()


def merge_communities(G: nx.Graph, communities: List[Set[int]]) -> nx.Graph:
    """
    Merge communities in the graph based on edge weights between them.

    Each community is merged into a single node, and edges are weighted
    by the sum of the weights of the edges between the communities.

    Parameters:
    - G: The original graph.
    - communities: A list of sets where each set represents a community of nodes.

    Returns:
    - A new graph where each community is a node, and edges are weighted
      by the sum of the edges between the communities.

    Author: Yang Tan
    """
    # Remove empty communities
    communities = remove_empty_communities(communities)

    # Create a new graph to store the merged communities
    new_G = nx.Graph()

    # Map each node to its community
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    # Add nodes to the new graph (one node per community)
    for i in range(len(communities)):
        new_G.add_node(i)

    # Add edges between communities based on the edges between the nodes
    for u, v, data in G.edges(data=True):
        community_u = node_to_community.get(u)
        community_v = node_to_community.get(v)

        if (
            community_u is not None
            and community_v is not None
            and community_u != community_v
        ):
            weight = data.get("weight", 1)

            if new_G.has_edge(community_u, community_v):
                new_G[community_u][community_v]["weight"] += weight
            else:
                new_G.add_edge(community_u, community_v, weight=weight)

    return new_G

def remove_empty_communities(communities: List[Set[int]]) -> List[Set[int]]:
    """
    Remove empty communities from the list of communities.

    Parameters:
    - communities: A list of sets where each set represents a community of nodes.

    Returns:
    - A new list of communities with empty communities removed.
    
    Author: Yang Tan
    """
    return [community for community in communities if len(community) > 0]
