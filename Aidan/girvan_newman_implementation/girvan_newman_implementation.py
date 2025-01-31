"""
** Run this file to run our Girvan-Newman implementation **

Wrapper file around our Girvan-Newman algorithm implementation to 
run our implementation and plot the result

Author: Aidan Roessler
"""

import networkx as nx
import girvan_newman as gn
import itertools
from typing import List, Set, Tuple, Dict

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
    network_x_girvan_newman_all_iterations = nx.community.girvan_newman(
        nx.karate_club_graph()
    )
    
    # print("network_x_girvan_newman_all_iterations:")
    # print(list(network_x_girvan_newman_all_iterations))

    network_x_iteration_with_highest_modularity = (
        find_network_x_communities_with_highest_modularity(
            immutable_G, network_x_girvan_newman_all_iterations
        )
    )

    print(network_x_iteration_with_highest_modularity)
    utils.plot_graph_with_communities(
        nx.karate_club_graph(),
        network_x_iteration_with_highest_modularity,
        title="Karate Club Graph with Communities Labeled by Network X's Girvan Newman Implementation",
    )

    print(
        f"Is my final result equal to Network X's? \n {girvan_newman_communities == list(network_x_iteration_with_highest_modularity)}"
    )


def find_network_x_communities_with_highest_modularity(
    G: nx.Graph, network_x_girvan_newman_communities_all_iterations
):
    max_modularity: float = 0.0
    communities_with_max_modularity: Tuple[Set[int]] = None

    for current_communities in network_x_girvan_newman_communities_all_iterations:
        current_modularity = nx.community.modularity(G, current_communities)

        if current_modularity > max_modularity:
            max_modularity = current_modularity
            communities_with_max_modularity = current_communities

    print(f"Final communities (Network X): {communities_with_max_modularity}")
    print(f"Final modularity (Network X): {max_modularity}")
    return communities_with_max_modularity

if __name__ == "__main__":
    main()
