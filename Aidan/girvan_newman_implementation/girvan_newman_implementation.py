"""
** Run this file to run our Girvan-Newman implementation **

Wrapper file around our Girvan-Newman algorithm implementation to 
run our implementation and plot the result

Author: Aidan Roessler
"""

import networkx as nx
import girvan_newman as gn
import itertools

import sys

sys.path.append("../../Util")
import utils


def main():
    """Main function that runs our implementation on test data for now"""

    # TODO: rethink name and if we should pass data in here
    # For now we are just passing in the karate club graph
    immutable_G = nx.karate_club_graph()
    G = nx.karate_club_graph()

    girvan_newman_communities = gn.girvan_newman(G)

    utils.plot_graph_with_communities(
        immutable_G,
        girvan_newman_communities,
        title="Karate Club Graph with Communities Labeled by our Girvan Newman Implementation",
    )

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html
    network_x_girvan_newman_communities_all_iterations = nx.community.girvan_newman(nx.karate_club_graph())
    print("network x communities")
    network_x_girvan_newman_limited_communities = list(
        itertools.takewhile(
            lambda c: len(c) <= len(girvan_newman_communities),
            network_x_girvan_newman_communities_all_iterations,
        )
    )[-1]

    print(network_x_girvan_newman_limited_communities)
    utils.plot_graph_with_communities(
        nx.karate_club_graph(),
        network_x_girvan_newman_limited_communities,
        title="Karate Club Graph with Communities Labeled by Network X's Girvan Newman Implementation",
    )

if __name__ == "__main__":
    main()
