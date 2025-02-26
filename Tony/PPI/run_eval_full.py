import time
import itertools
import csv
import math
import os
import psutil
import random
import numpy as np
import networkx as nx
import pandas as pd
import tracemalloc
from networkx.algorithms.community.quality import modularity
from typing import List, Set, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

def measure_performance(algorithm, graph, kmax=3):
    """
    Measures the performance (runtime and peak memory usage) of the given algorithm.
    Uses tracemalloc to capture the peak memory usage during execution.
    If the algorithm is run_bvns, it passes an extra parameter kmax.
    """
    tracemalloc.start()
    start_time = time.time()
    
    # For BVNS, pass kmax; otherwise, just call with graph.
    if algorithm.__name__ == "run_bvns":
        partition = algorithm(graph, kmax)
    else:
        partition = algorithm(graph)
    
    elapsed_time = time.time() - start_time
    # Get the current and peak memory usage (in bytes) during the tracing period.
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_used = peak  # Using the peak memory usage as the metric.
    return partition, elapsed_time, memory_used

# -------------------------------
# Community detection algorithms
# -------------------------------
def run_bvns(G: nx.Graph, kmax=3):
    import sys
    sys.path.append("../../Jake/bvns")
    import bvns_implementation as bvns
    return [set(lst) for lst in bvns.bvns_with_communities_specified(G)]

def run_louvain_nx(G: nx.Graph):
    from networkx.algorithms.community import louvain_communities
    c = louvain_communities(G, weight=None, resolution=1, threshold=1e-7, seed=None)
    return list(c)

def run_girvan_newman_nx(G: nx.Graph):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return list(gn.find_network_x_communities_with_highest_modularity(G, G.number_of_nodes()))

def run_girvan_newman(G: nx.Graph):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return gn.main(G, False)

def run_label_propagation(G: nx.Graph):
    from networkx.algorithms.community import label_propagation_communities
    return list(label_propagation_communities(G))

def run_asyn_lpa(G: nx.Graph):
    from networkx.algorithms.community import asyn_lpa_communities
    return list(asyn_lpa_communities(G, weight=None, seed=None))

def run_louvain(G: nx.Graph):
    import sys
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm
    return lm.main(G)

# -------------------------------
# Data preparation and loading
# -------------------------------
def prepare_network(input_file: str, output_file: str, filters: Dict[str, float]) -> None:
    df = pd.read_csv(input_file, sep=r'\s+')
    mask = pd.Series(True, index=df.index)
    for col, threshold in filters.items():
        mask &= (df[col] >= threshold)
    df_filtered = df[mask].copy()
    df_filtered['protein1'] = df_filtered['protein1'].str.replace(r'^4932\.', '', regex=True)
    df_filtered['protein2'] = df_filtered['protein2'].str.replace(r'^4932\.', '', regex=True)
    df_filtered[['protein1', 'protein2']].to_csv(output_file, header=False, index=False, sep=' ')

def load_data(file_path: str, data_type: str = 'network') -> Union[nx.Graph, List[Set]]:
    if data_type == 'network':
        return nx.read_edgelist(file_path, comments='#', nodetype=str)
    elif data_type == 'clusters':
        clusters = []
        with open(file_path) as f:
            clusters = [set(line.strip().split()) for line in f]
        return clusters

# -------------------------------
# Clustering evaluation
# -------------------------------
def evaluate_clustering(pred_clusters: List[Set], 
                        ref_clusters: List[Set], 
                        overlap_threshold: float = 0.3) -> Dict[str, float]:
    def calc_overlap_metrics(clusters1: List[Set], clusters2: List[Set], threshold: float) -> float:
        hits = 0
        for c1 in clusters1:
            for c2 in clusters2:
                intersection = len(c1.intersection(c2))
                union_size = len(c1) + len(c2)
                if union_size > 0 and (intersection / union_size) > threshold:
                    hits += 1
                    break
        return hits / len(clusters1) if clusters1 else 0.0

    def calc_value_metrics(clusters1: List[Set], clusters2: List[Set]) -> float:
        total_nodes = sum(len(c) for c in clusters1)
        if not total_nodes:
            return 0.0
        sum_intersect = 0
        for c1 in clusters1:
            best = max(len(c1 & c2) for c2 in clusters2) if clusters2 else 0
            sum_intersect += best
        return sum_intersect / total_nodes

    def calc_separation(p_clusters: List[Set], r_clusters: List[Set]) -> float:
        if not (p_clusters and r_clusters):
            return 0.0
        def sum_intersections(c1s: List[Set], c2s: List[Set]) -> float:
            return sum(
                (len(c1 & c2)**2) / (len(c1) * len(c2))
                for c1 in c1s for c2 in c2s
                if len(c1) > 0 and len(c2) > 0
            )
        sum1 = sum_intersections(p_clusters, r_clusters)
        sum2 = sum_intersections(r_clusters, p_clusters)
        denom = len(p_clusters) * len(r_clusters)
        return np.sqrt((sum1 * sum2) / (denom**2)) if denom > 0 else 0.0

    recall = calc_overlap_metrics(ref_clusters, pred_clusters, overlap_threshold)
    precision = calc_overlap_metrics(pred_clusters, ref_clusters, overlap_threshold)
    f_measure = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    ppv = calc_value_metrics(pred_clusters, ref_clusters)
    sn = calc_value_metrics(ref_clusters, pred_clusters)
    acc = np.sqrt(ppv * sn)
    sep = calc_separation(pred_clusters, ref_clusters)

    return {
        "Recall": recall,
        "Precision": precision,
        "F-measure": f_measure,
        "PPV": ppv,
        "SN": sn,
        "ACC": acc,
        "SEP": sep,
        "NumClusters": len(pred_clusters)
    }

# -------------------------------
# Main execution: run all methods in parallel
# -------------------------------
def main():
    input_file = "4932.protein.links.full.v12.0.txt"
    network_file = "yeast_network.txt"
    reference_file = "CYC2008.txt"

    # Prepare the network from the input file.
    prepare_network(
        input_file=input_file,
        output_file=network_file,
        filters={"experiments":0}
    )
    print(f"Input network saved to: {network_file}")

    # Load the network graph.
    G = load_data(network_file, 'network')
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load the reference clusters.
    reference_clusters = load_data(reference_file, 'clusters')
    print(f"Loaded reference data which has: {len(reference_clusters)} complexes")

    # Define all community detection methods to run.
    COMMUNITY_METHODS = {
        'girvan_newman': run_girvan_newman,
        'girvan_newman_nx': run_girvan_newman_nx,
        'louvain': run_louvain,
        'louvain_nx': run_louvain_nx,
        'bvns': run_bvns,
        'label_propagation': run_label_propagation,
        'asyn_lpa': run_asyn_lpa,
    }

    results = {}
    # Run all methods in parallel.
    with ProcessPoolExecutor() as executor:
        future_to_method = {}
        for method_name, method_func in COMMUNITY_METHODS.items():
            # Submit each method wrapped in measure_performance.
            future = executor.submit(measure_performance, method_func, G)
            future_to_method[future] = method_name
            print("submitted",method_name)

        for future in as_completed(future_to_method):
            method_name = future_to_method[future]
            try:
                partition, elapsed, memory_used = future.result()
            except Exception as exc:
                print(f"Error running {method_name}: {exc}")
                continue

            # Evaluate the clustering performance.
            metrics = evaluate_clustering(partition, reference_clusters)
            results[method_name] = {
                "partition": partition,
                "elapsed": elapsed,
                "memory_used": memory_used,
                "metrics": metrics
            }
            print(f"Method {method_name}: {metrics['NumClusters']} communities found in {elapsed:.2f}s, peak memory: {memory_used} bytes")

    # Prepare CSV output.
    csv_columns = [
        "Method", "NumClusters", 
        "Recall", "Precision", "F-measure", "PPV", "SN", "ACC", "SEP",
        "Runtime", "PeakMemory"
    ]
    csv_output = "results_0.csv"
    with open(csv_output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_columns)
        writer.writeheader()
        for method_name, data in results.items():
            metrics = data["metrics"]
            writer.writerow({
                "Method": method_name,
                "NumClusters": metrics["NumClusters"],
                "Recall": f"{metrics['Recall']:.3f}",
                "Precision": f"{metrics['Precision']:.3f}",
                "F-measure": f"{metrics['F-measure']:.3f}",
                "PPV": f"{metrics['PPV']:.3f}",
                "SN": f"{metrics['SN']:.3f}",
                "ACC": f"{metrics['ACC']:.3f}",
                "SEP": f"{metrics['SEP']:.3f}",
                "Runtime": f"{data['elapsed']:.3f}",
                "PeakMemory": data["memory_used"]
            })

    print(f"\nAll done. Results saved in '{csv_output}'.")

if __name__ == "__main__":
    main()