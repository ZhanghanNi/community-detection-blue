"""
File to run all of our implementations on all of our datasets

Authors: Aidan Roessler, Tony Ni
"""

import networkx as nx
import sys
import argparse

sys.path.append("Util")
sys.path.append("Aidan/girvan_newman_implementation")
sys.path.append("Yang_Tan")
sys.path.append("Tony/PPI")
sys.path.append("Jake")
import girvan_newman_implementation as gn
import louvain_method_implementation as lm
import bvns_implementation as bvns
import run_eval

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
        "dataset", choices=["karate_club", "college_football"], help="The dataset to run the selected algorithm on"
    )

    args = parser.parse_args()
    dataset: nx.Graph
    dataset_name: str
    bvns_kmax: int

    match args.dataset:
        case "karate_club":
            dataset_name = "Karate Club"
            dataset = nx.karate_club_graph()
            bvns_kmax = 5
        case "college_football":
            dataset_name = "College Football"
            dataset = nx.read_gml("Jake/football/football.gml")
            bvns_kmax = 4

    match args.algorithm:
        case "girvan_newman":
            gn.main(dataset, weighted=nx.is_weighted(dataset), dataset_name=dataset_name)
        case "louvain_method":
            lm.main(dataset, dataset_name=dataset_name)
        case "bvns":
            bvns.main(dataset, kmax=bvns_kmax, dataset_name=dataset_name)

    # run_eval.main()

if __name__ == "__main__":
    main()
