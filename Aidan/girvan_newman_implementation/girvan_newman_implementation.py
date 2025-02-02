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


def main(G: nx.Graph, weighted: bool):
    """Main function that runs our implementation on test data for now"""
    # TODO: rethink name and if we should pass data in here
    if not weighted:
        # Remove edges from the weighted version of the graph
        weighted_g = G

        for edge in weighted_g.edges(data=True):
            edge[2]['weight'] = 1

        immutable_G = weighted_g
        G_to_pass = immutable_G.copy()
    else:
        immutable_G = G
        G_to_pass = G.copy()

    girvan_newman_communities = gn.girvan_newman(G_to_pass, immutable_G, False)

    utils.plot_graph_with_communities(
        immutable_G,
        girvan_newman_communities,
        title=f"{"Weighted" if weighted else "Unweighted"} Karate Club Graph with Communities Labeled by our Girvan Newman Implementation",
        label_edges=weighted
    )

    network_x_iteration_with_highest_modularity = (
        find_network_x_communities_with_highest_modularity(
            immutable_G, immutable_G.number_of_nodes()
        )
    )

    utils.plot_graph_with_communities(
        nx.karate_club_graph(),
        network_x_iteration_with_highest_modularity,
        title=f"{"Weighted" if weighted else "Unweighted"} Karate Club Graph with Communities Labeled by Network X's Girvan Newman Implementation",
        label_edges=weighted,
    )

    print(
        f"Is our final result equal to Network X's? \n {girvan_newman_communities == list(network_x_iteration_with_highest_modularity)}"
    )


def find_network_x_communities_with_highest_modularity(G: nx.Graph, max_communities: int):
    max_modularity: float = -1.0
    communities_with_max_modularity: Tuple[Set[int]] = None

    # Generate all possible community partitions using the Girvan–Newman algorithm
    comp = nx.community.girvan_newman(G)

    # Use itertools.takewhile to iterate over partitions until the number of communities exceeds max_communities
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.centrality.girvan_newman.html
    limited = itertools.takewhile(lambda c: len(c) <= max_communities, comp)

    for current_communities in limited:        
        current_modularity = utils.modularity(G, list(current_communities))

        if current_modularity > max_modularity:
            max_modularity = current_modularity
            communities_with_max_modularity = current_communities

    print(f"Final communities (NetworkX): {communities_with_max_modularity}")
    print(f"Final modularity (NetworkX): {max_modularity}")
    return communities_with_max_modularity

if __name__ == "__main__":
    main()
