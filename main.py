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
sys.path.append("Yang_Tan/simulation")
sys.path.append("Tony/PPI")
sys.path.append("Jake/bvns")
sys.path.append("Jake/football")
sys.path.append("Jake/neural_connectivity")
import girvan_newman_implementation as gn
import louvain_method_implementation as lm
import bvns_implementation as bvns
import undirected_neural_data
import trajectory_graph_generator
import trajectory_experiment


def main():
    """
    Highest level function in the whole project that takes in arguments from the CLI to
    run one of our algorithms on one of our datasets based on user input.

    Author: Aidan Roessler
    """
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Run our implementations of community detection algorithms on a dataset."
    )
    parser.add_argument(
        "--algorithm",
        choices=["girvan_newman", "louvain_method", "bvns"],
        required=True,
        help="The implementation of a community detection algorithm to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "karate_club",
            "college_football",
            "celegans_neural",
            "urban_movement_synthetic",
        ],
        required=True,
        help="The dataset to run the selected algorithm on.",
    )

    args = parser.parse_args()
    dataset: nx.Graph
    dataset_name: str
    algorithm_name: str
    bvns_kmax: int = 3
    generated_communities = []
    subareas_for_urban_movement = []

    if args.dataset == "karate_club":
        dataset_name = "Karate Club"
        dataset = nx.karate_club_graph()
        bvns_kmax = 5
    if args.dataset == "college_football":
        dataset_name = "College Football"
        dataset = nx.read_gml("Jake/football/football.gml")
        bvns_kmax = 4
    if args.dataset == "celegans_neural":
        dataset_name = "Neural network of the nematode C. Elegans"
        dataset = undirected_neural_data.get_neuronal_connectivity_graph(
            "Jake/neural_connectivity/undirected_added_nodes.csv"
        )
    if args.dataset == "urban_movement_synthetic":
        dataset_name = "Simulated Urban Movement Data (Merged Random Trajectories)"
        # Use the merged trajectory graph
        dataset, subareas_for_urban_movement = trajectory_graph_generator.main()

    print(f"Number of nodes: {dataset.number_of_nodes()}")
    print(f"Number of edges: {dataset.number_of_edges()}")

    match args.algorithm:
        case "girvan_newman":
            algorithm_name = "Girvan Newman"
            print(f"Running {algorithm_name} on {dataset_name}")

            generated_communities = gn.main(
                dataset, weighted=nx.is_weighted(dataset), dataset_name=dataset_name
            )

        case "louvain_method":
            algorithm_name = "Louvain Method"
            print(f"Running {algorithm_name} on {dataset_name}")

            generated_communities = lm.main(dataset, dataset_name=dataset_name)

        case "bvns":
            algorithm_name = "BVNS"
            print(f"Running {algorithm_name} on {dataset_name}")

            generated_communities = bvns.main(
                dataset, kmax=bvns_kmax, dataset_name=dataset_name
            )

    # Plot additional visualizations for urban movement that are dependent on communities
    if args.dataset == "urban_movement_synthetic":
        # a and b parameters are the x-axis and y-axis ranges
        result_analysis = trajectory_experiment.Result_Analysis(
            dataset,
            generated_communities,
            subareas_for_urban_movement,
            100,
            100,
            algorithm_name,
        )
        result_analysis.plot_results()


if __name__ == "__main__":
    main()
