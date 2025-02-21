import time
import os
import psutil
import networkx as nx
from networkx.algorithms.community.quality import modularity
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import argparse
import sys


def run_girvan_newman(graph):
    sys.path.append("../girvan_newman_implementation")
    import girvan_newman_implementation as gn

    return gn.main(graph, nx.is_weighted(graph), benchmarking_mode=True)


def run_louvain(graph):
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm

    return lm.main(graph, benchmarking_mode=True)

def run_bvns(graph, kmax=3):
    sys.path.append("../../Jake/bvns")
    import bvns_implementation as bvns

    return bvns.bvns(graph, kmax=kmax)

def measure_performance(algorithm, graph, kmax=3):
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    start_time = time.time()
    if algorithm == run_bvns:
        partition = algorithm(graph, kmax)
    else:
        partition = algorithm(graph)
    elapsed_time = time.time() - start_time
    memory_after = process.memory_info().rss
    memory_used = memory_after - memory_before
    return partition, elapsed_time, memory_used


def run_benchmark(graph: nx.Graph, kmax=3):
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
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Run our implementations of community detection algorithms on a dataset."
    )
    parser.add_argument(
        "dataset",
        choices=["karate_club", "college_football", "celegans_neural"],
        help="The dataset to run the selected algorithm on",
    )
    args = parser.parse_args()
    dataset: nx.Graph
    bvns_kmax = 3

    if args.dataset == "karate_club":
        dataset = nx.karate_club_graph()
        bvns_kmax = 4
    if args.dataset == "college_football":
        dataset = nx.read_gml("../../Jake/football/football.gml")
        bvns_kmax = 4
    if args.dataset == "celegans_neural":
        sys.path.append("../../Jake/neural_connectivity")
        sys.path.append("../../Jake/bvns")  # Ensure bvns is on the path

        import csv_to_nx_graph as csv_to_nx

        dataset = csv_to_nx.get_neuronal_connectivity_graph(
            "../../Jake/neural_connectivity/SI7_herm.csv"
        )

    output_file = f"{args.dataset}_benchmark_results.csv"

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
            except Exception as e:
                print(f"Error in benchmark execution: {e}")
