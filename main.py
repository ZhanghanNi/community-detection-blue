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
import girvan_newman_implementation as gn
import louvain_method_implementation as lm
import run_eval

def main():
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Run our implementations of community detection algorithms on a dataset."
    )
    parser.add_argument(
        "algorithm",
        choices=["girvan_newman", "louvain_method"],
        help="The implementation of a community detection algorithm to run.",
    )
    parser.add_argument(
        "dataset", choices=["karate_club"], help="The dataset to run the selected algorithm on"
    )

    args = parser.parse_args()
    dataset: nx.Graph

    match args.dataset:
        case "karate_club":
            dataset = nx.karate_club_graph()

    match args.algorithm:
        case "girvan_newman":
            label = "Karate Club"
            gn.main(dataset, weighted=True, dataset_name=label)
            gn.main(dataset, weighted=False, dataset_name=label)
        case "louvain_method":
            lm.main(dataset, dataset_name="Karate Club")

    run_eval.main()


if __name__ == "__main__":
    main()
