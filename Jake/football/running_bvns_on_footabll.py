import sys
sys.path.append("../../Util")
sys.path.append("../bvns")
import utils
import bvns_implementation as bvns
import networkx as nx

def main():
    graph = nx.read_gml("./football.gml")
    solution = bvns.bvns(graph, 4)
    print("modularity: ", utils.modularity(graph, solution))
    utils.plot_graph_with_communities(graph, solution)

if __name__ == "__main__":
    main()