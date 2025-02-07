import sys
sys.path.append("../Util")
import utils
import bvns_implementation as bvns
import networkx as nx

def main():
    # Visualize it on the karate graph
    karate_graph =nx.karate_club_graph()
    bvns_result = bvns.bvns(karate_graph, 5)
    utils.plot_graph_with_communities(karate_graph, bvns_result)

if __name__ == "__main__":
    main()