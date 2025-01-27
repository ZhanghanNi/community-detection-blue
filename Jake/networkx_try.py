import networkx as nx

# Load the GML file
graph = nx.read_gml("./football/football.gml")

# Print information about the graph
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())
print("Nodes:", list(graph.nodes()))
# print("Edges:", list(graph.edges()))
