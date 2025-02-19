import networkx as nx
#SBM_2blocks_400nodes_pin0.010_pout0.001

g = nx.stochastic_block_model([400,400], [[0.01,0.001],[0.001,0.01]])
g = nx.stochastic_block_model([20,20], [[0.001,0.001],[0.001,0.001]])

#SBM_2blocks_20nodes_pin0.020_pout0.005
import matplotlib.pyplot as plt
#nx.draw(g)
#plt.savefig("bug.png")

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
    plt.savefig("bug_2.png")


plot_graph(g)
print(nx.is_connected(g))

communities = list(nx.community.girvan_newman(g))

print(communities)

def run_girvan_newman(G):
    """
    Uses your custom Girvan–Newman implementation.
    """
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return gn.main(G, False)


communities = run_girvan_newman(g)

print(communities)