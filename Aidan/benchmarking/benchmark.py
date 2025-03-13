"""
File to run all of our algorithms on all of our datasets (minus PPI + synthetic datasets)
and collect metrics for us to analyze. 

Author: Aidan Roessler, Tony Ni
"""

import time
import networkx as nx
from networkx.algorithms.community.quality import modularity
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import argparse
import sys
import tracemalloc

sys.path.append("../../Yang_Tan/simulation")
import trajectory_experiment


def run_girvan_newman(graph):
    """
    Runs our Girvan-Newman implementation.

    Parameters:
    - graph: NetworkX Graph on which the Girvan-Newman algorithm will be run

    Returns:
    - A list of communities found by Girvan-Newman (represented as sets)
    """
    sys.path.append("../girvan_newman_implementation")
    import girvan_newman_implementation as gn

    return gn.main(graph, nx.is_weighted(graph), benchmarking_mode=True)


def run_louvain(graph):
    """
    Runs our Louvain method implementation.

    Parameters:
    - graph: NetworkX Graph on which the Louvain method will be run

    Returns:
    - A list of communities found by the Louvain method (represented as sets)
    """
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm

    return lm.main(graph, benchmarking_mode=True)


def run_bvns(graph, kmax=3):
    """
    Runs our Basic Variable Neighborhood Search (BVNS) implementation.

    Parameters:
    - graph: NetworkX Graph on which the BVNS algorithm will be run
    - kmax: The maximum k-value used in BVNS (optional, default is 3)

    Returns:
    - A list of communities found by BVNS (represented as sets)
    """
    sys.path.append("../../Jake/bvns")
    import bvns_implementation as bvns

    return bvns.bvns_with_communities_specified(graph, kmax=kmax)


def measure_performance(algorithm, graph, kmax=3):
    """
    Measures the time and memory usage of a given community detection algorithm.

    Parameters:
    - algorithm: The function that implements the community detection algorithm
    - graph: NetworkX Graph on which the algorithm will be run
    - kmax: The maximum k-value if BVNS is selected (optional, default is 3)

    Returns:
    - partition: The resulting communities from the algorithm
    - elapsed_time: The total time taken by the algorithm
    - memory_used: The peak memory usage during execution (in bytes)
    """    
    tracemalloc.start()
    start_time = time.time()
    
    if algorithm == run_bvns:
        partition = algorithm(graph, kmax)
    else:
        partition = algorithm(graph)
    
    elapsed_time = time.time() - start_time
    # Get the current and peak memory usage (in bytes) during the tracing period.
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_used = peak  # Using the peak memory usage as the metric.
    return partition, elapsed_time, memory_used

def run_benchmark(graph: nx.Graph, kmax=3):
    """
    Runs all three community detection algorithms (Girvan-Newman, Louvain, BVNS)
    on the provided graph and collects performance metrics.

    Parameters:
    - graph: NetworkX Graph to be used for benchmarking
    - kmax: The maximum k-value for BVNS (optional, default is 3)

    Returns:
    - A dictionary containing:
        - Number of nodes/edges
        - Execution time, peak memory usage, and modularity for each algorithm
        - The final communities outputted by each algorithm
    """    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Initialize results
    results = {
        "NumNodes": num_nodes,
        "NumEdges": num_edges,
        "GN_Time": None,
        "GN_Memory": None,
        "GN_Modularity": None,
        "GN_Final_Communities": None,
        "LV_Time": None,
        "LV_Memory": None,
        "LV_Modularity": None,
        "LV_Final_Communities": None,
        "BVNS_Time": None,
        "BVNS_Memory": None,
        "BVNS_Modularity": None,
        "BVNS_Final_Communities": None,
    }

    # Run Girvan-Newman
    try:
        gn_partition, gn_time, gn_memory = measure_performance(run_girvan_newman, graph)
        results.update(
            {
                "GN_Time": gn_time,
                "GN_Memory": gn_memory,
                "GN_Modularity": modularity(graph, gn_partition),
                "GN_Final_Communities": gn_partition,
            }
        )
    except Exception as e:
        print(f"Error running Girvan-Newman: {e}")

    # Run Louvain
    try:
        lv_partition, lv_time, lv_memory = measure_performance(run_louvain, graph)
        results.update(
            {
                "LV_Time": lv_time,
                "LV_Memory": lv_memory,
                "LV_Modularity": modularity(graph, lv_partition),
                "LV_Final_Communities": lv_partition,
            }
        )
    except Exception as e:
        print(f"Error running Louvain: {e}")

    # Run BVNS
    try:
        bvns_partition, bvns_time, bvns_memory = measure_performance(
            run_bvns, graph, kmax=kmax
        )
        results.update(
            {
                "BVNS_Time": bvns_time,
                "BVNS_Memory": bvns_memory,
                "BVNS_Modularity": modularity(graph, bvns_partition),
                "BVNS_Final_Communities": bvns_partition,
            }
        )
    except Exception as e:
        print(f"Error running BVNS: {e}")

    return results


if __name__ == "__main__":
    """
    Runs our full benchmarking pipeline on the dataset the user specifies
    """
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Run our implementations of community detection algorithms on a dataset."
    )
    parser.add_argument(
        "dataset",
        choices=["karate_club", "college_football", "celegans_neural", "urban_movement_synthetic"],
        help="The dataset to run the selected algorithm on",
    )
    args = parser.parse_args()
    dataset: nx.Graph
    bvns_kmax = 3
    subareas_for_urban_movement = []

    if args.dataset == "karate_club":
        dataset = nx.karate_club_graph()
        bvns_kmax = 4
    if args.dataset == "college_football":
        dataset = nx.read_gml("../../Jake/football/football.gml")
        bvns_kmax = 4
    if args.dataset == "celegans_neural":
        sys.path.append("../../Jake/neural_connectivity")
        sys.path.append("../../Jake/bvns")  # Ensure bvns is on the path

        import undirected_neural_data

        dataset = undirected_neural_data.get_neuronal_connectivity_graph(
            "../../Jake/neural_connectivity/undirected_added_nodes.csv"
        )
    if args.dataset == "urban_movement_synthetic":
        sys.path.append("Yang_Tan/simulation")
        import trajectory_graph_generator

        dataset, subareas_for_urban_movement = trajectory_graph_generator.main()

    output_file = f"final_{args.dataset}_benchmark_results.csv"

    fieldnames = [
        "NumNodes",
        "NumEdges",
        "GN_Time",
        "GN_Memory",
        "GN_Modularity",
        "GN_Final_Communities",
        "LV_Time",
        "LV_Memory",
        "LV_Modularity",
        "LV_Final_Communities",
        "BVNS_Time",
        "BVNS_Memory",
        "BVNS_Modularity",
        "BVNS_Final_Communities",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=2) as executor:
            future = executor.submit(run_benchmark, dataset, bvns_kmax)
            try:
                result = future.result()
                writer.writerow(result)
                print("Benchmarking complete. Results saved to", output_file)
                # Visualize Final Results for Urban Movement synthetic data
                if args.dataset == "urban_movement_synthetic":
                    print("Visualizing Urban Movement Results")
                    girvan_newman_result_analysis = (
                        trajectory_experiment.Result_Analysis(
                            dataset,
                            result["GN_Final_Communities"],
                            subareas_for_urban_movement,
                            100,
                            100,
                            "Girvan Newman",
                        )
                    )
                    girvan_newman_result_analysis.plot_results()

                    louvain_result_analysis = trajectory_experiment.Result_Analysis(
                        dataset,
                        result["LV_Final_Communities"],
                        subareas_for_urban_movement,
                        100,
                        100,
                        "Louvain",
                    )
                    louvain_result_analysis.plot_results()

                    bvns_result_analysis = trajectory_experiment.Result_Analysis(
                        dataset,
                        result["BVNS_Final_Communities"],
                        subareas_for_urban_movement,
                        100,
                        100,
                        "BVNS",
                    )
                    bvns_result_analysis.plot_results()

            except Exception as e:
                print(f"Error in benchmark execution: {e}")
