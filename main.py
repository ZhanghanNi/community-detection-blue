"""
File to run all of our implementations on all of our datasets

Authors: Aidan Roessler, Tony Ni
"""

import networkx as nx
import sys
import argparse
from typing import List, Set

sys.path.append("Util")
sys.path.append("Aidan/girvan_newman_implementation")
sys.path.append("Yang_Tan")
sys.path.append("Tony/PPI")
sys.path.append("Jake")
import girvan_newman_implementation as gn
import louvain_method_implementation as lm
import bvns_implementation as bvns
import run_eval


karate_club_ground_truth_communities: List[Set[int]] = [
    {3, 4, 2, 1, 8, 22, 20, 18, 13, 12, 7, 17, 6, 5, 11},
    {29, 25, 28, 33, 34, 30, 24, 31, 9, 23, 21, 19, 16, 15, 26, 32, 27, 10},
]

karate_club_ground_truth_1, karate_club_ground_truth_2 = (
    karate_club_ground_truth_communities
)


def main():
    """
    Highest level function in the whole project that takes in arguments from the CLI to
    run one of our algorithms on one of our datasets based on user input.
    """
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Run our implementations of community detection algorithms on a dataset."
    )
    parser.add_argument(
        "algorithm",
        choices=["girvan_newman", "louvain_method", "bvns"],
        help="The implementation of a community detection algorithm to run.",
    )
    parser.add_argument(
        "dataset",
        choices=["karate_club", "college_football", "celegans_neural"],
        help="The dataset to run the selected algorithm on",
    )

    args = parser.parse_args()
    dataset: nx.Graph
    dataset_name: str
    bvns_kmax: int = 3

    match args.dataset:
        case "karate_club":
            dataset_name = "Karate Club"
            dataset = nx.karate_club_graph()
            bvns_kmax = 5
        case "college_football":
            dataset_name = "College Football"
            dataset = nx.read_gml("Jake/football/football.gml")
            bvns_kmax = 4
        case "celegans_neural":
            dataset_name = "Neural network of the nematode C. Elegans"
            dataset = nx.read_gml("Jake/celegansneural/celegansneural.gml")

    match args.algorithm:
        case "girvan_newman":
            gn.main(
                dataset, weighted=nx.is_weighted(dataset), dataset_name=dataset_name
            )
            # Only run until there are two communities to compare to the 2 real life communities
            if dataset_name == "Karate Club":
                generated_communities = gn.main(
                    dataset,
                    weighted=nx.is_weighted(dataset),
                    dataset_name=dataset_name,
                    max_communities=2,
                )
                # Since Girvan and Newman used 1 indexing and we use 0, shift our results by one
                shifted_generated_1 = {n + 1 for n in generated_communities[0]}
                shifted_generated_2 = {n + 1 for n in generated_communities[1]}

                incorrectly_assigned_nodes = (
                    karate_club_ground_truth_1 & shifted_generated_2
                ) | (karate_club_ground_truth_2 & shifted_generated_1)
                print(
                    "Incorrectly assigned nodes when amount of communities = 2:",
                    incorrectly_assigned_nodes,
                )

        case "louvain_method":
            lm.main(dataset, dataset_name=dataset_name)
        case "bvns":
            bvns.main(dataset, kmax=bvns_kmax, dataset_name=dataset_name)

    # run_eval.main()


if __name__ == "__main__":
    main()
