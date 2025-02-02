import networkx as nx
from eval_utils import *
def main():
    input_file = "Tony/PPI/4932.protein.links.full.v12.0.txt"
    network_file = "Tony/PPI/yeast_network.txt"

    prepare_network(
        input_file=input_file,
        output_file=network_file,
        filters={"experiments": 229}
    )
    print(f"Input network saved to: {network_file}")

    G = load_data(network_file, 'network')

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    reference_clusters = load_data("Tony/PPI/CYC2008.txt", 'clusters')
    print(f"Loaded reference data which has: {len(reference_clusters)} complexes")

    #Community Detection and Eval
    methods = ['girvan_newman', 'louvain']

    for method in methods:
        print(f"\nRunning {method.replace('_', '-')}...")
        communities = detect_communities(G, method)
        print(f"Found {len(communities)} communities")
        metrics = evaluate_clustering(communities, reference_clusters)
        print(f"{method.replace('_', '-')} metrics vs reference:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()