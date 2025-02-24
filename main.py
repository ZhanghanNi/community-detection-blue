"""
File to run all of our implementations on all of our datasets

Authors: Aidan Roessler, Tony Ni
"""

import networkx as nx
import sys
import argparse
from typing import List, Set, Dict, Union

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
import comparing_to_football_results as football_results
import undirected_neural_data
import trajectory_graph_generator
import trajectory_experiment
import run_eval

Node = Union[int, str]

karate_club_ground_truth_communities: List[Set[int]] = [
    {3, 4, 2, 1, 8, 22, 20, 18, 13, 12, 7, 17, 6, 5, 11},
    {29, 25, 28, 33, 34, 30, 24, 31, 9, 23, 21, 19, 16, 15, 26, 32, 27, 10},
]

# Since Girvan and Newman used 1 indexing and we use 0, shift their results by one
karate_club_ground_truth_1 = {n - 1 for n in karate_club_ground_truth_communities[0]}
karate_club_ground_truth_2 = {n - 1 for n in karate_club_ground_truth_communities[1]}


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
        choices=["karate_club", "college_football", "celegans_neural", "urban_movement_synthetic"],
        help="The dataset to run the selected algorithm on",
    )

    args = parser.parse_args()
    dataset: nx.Graph
    dataset_name: str
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
            generated_communities = gn.main(
                dataset, weighted=nx.is_weighted(dataset), dataset_name=dataset_name
            )
            if dataset_name == "Karate Club":
                # Only run until there are two communities to compare to the 2 real life communities
                generated_1, generated_2 = gn.main(
                    dataset,
                    weighted=nx.is_weighted(dataset),
                    dataset_name=dataset_name,
                    max_communities=2,
                )

                incorrectly_assigned_nodes = (
                    karate_club_ground_truth_1 & generated_2
                ) | (karate_club_ground_truth_2 & generated_1)

                print(
                    "Incorrectly assigned nodes when amount of communities = 2:",
                    incorrectly_assigned_nodes,
                )

            if dataset_name == "College Football":
                college_football_ground_truth_communities = (
                    football_results.find_football_truth_values()
                )

                ten_communities = gn.main(
                    dataset, weighted=nx.is_weighted(dataset), dataset_name=dataset_name
                )

                print(
                    f"Does our implementation return the same communities as the ground truth? {college_football_ground_truth_communities == ten_communities}"
                )

        case "louvain_method":
            generated_communities = lm.main(dataset, dataset_name=dataset_name)

            if dataset_name == "Karate Club":
                network_x_communities = nx.community.louvain_communities(
                    nx.karate_club_graph(), seed=1234
                )

                print(f"Final Communities (Network X): {network_x_communities}")

                print(
                    f"Is our final result equal to Network X's? \n {generated_communities == network_x_communities}"
                )

        case "bvns":
            generated_communities = bvns.main(dataset, kmax=bvns_kmax, dataset_name=dataset_name)
    
    # Plot additional visualizations for urban movement that are dependent on communities
    if args.dataset == "urban_movement_synthetic":
        # Last two parameters here are a and b, the x-axis and y-axis ranges
        result_analysis = trajectory_experiment.Result_Analysis(dataset, generated_communities, subareas_for_urban_movement, 100, 100)
        result_analysis.plot_results()

if __name__ == "__main__":
    main()
