import matplotlib.pyplot as plt
import networkx as nx


# Create the Karate Club graph
G = nx.karate_club_graph()
print("Edge Weights:")
for u, v, data in G.edges(data=True):
    print(f"({u}, {v}) - weight: {data['weight']}")
# Assign weights to edges (for example, set all weights to 1)
for u, v in G.edges():
    G[u][v]['weight'] = 1

# Print the edge weights
print("Edge Weights:")
for u, v, data in G.edges(data=True):
    print(f"({u}, {v}) - weight: {data['weight']}")

# Draw the graph using a circular layout
nx.draw_circular(G, with_labels=True, node_color='lightblue', edge_color='gray')

# Show the plot
plt.title("Karate Club Graph")
plt.show()