import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import louvain_method_copy as lm
from typing import List, Set, Tuple


def plot_graph(G: nx.Graph, communities: List[Set[int]], label_edges: bool = False):
    """
    Plot the graph with nodes colored according to their communities.
    
    Parameters:
    - G: The graph.
    - communities: The list of communities.
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
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.rainbow, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Optionally label edges with their weights
    if label_edges:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Create legend for communities
    community_patches = []
    for idx, community in enumerate(communities):
        # Each community will have a unique color
        patch = mpatches.Patch(color=plt.cm.rainbow(idx / len(communities)), label=f"Community {idx + 1}")
        community_patches.append(patch)
    
    # Adjust the position of the legend to avoid covering the graph
    plt.legend(handles=community_patches, loc='upper left', bbox_to_anchor=(1, 1), title="Communities")
    
    # Adjust layout to make space for the legend
    plt.tight_layout()
    
    # Show the plot
    plt.title("Graph with Communities Detected by Louvain Method")
    plt.show()


def main():
    G = nx.karate_club_graph()

    print("Edge Weights:")
    for u, v, data in G.edges(data=True):
        print(f"({u}, {v}) - weight: {data['weight']}")
    # Apply Louvain method
    
    
    # You can set a random seed here so it might be useful to do so
    communities = nx.community.louvain_communities(G)

    # Print the detected communities
    print("\nDetected communities:")
    for i, community in enumerate(communities):
        print(f"Community {i + 1}: {community}")
    
    # Plot the graph with communities
    plot_graph(G, communities)
    merged_G = lm.merge_communities(G, communities)
    print("\nAfter merging:")
    plot_graph(merged_G, communities = [{node} for node in merged_G.nodes()],label_edges = True)

if __name__ == "__main__":
    main()