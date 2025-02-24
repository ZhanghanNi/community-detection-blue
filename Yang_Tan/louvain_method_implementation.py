import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import louvain_method as lm
from typing import List, Set, Tuple

import sys
sys.path.append("../Util")
import utils

def main(G: nx.Graph, dataset_name: str = "Graph", benchmarking_mode: bool = False)-> List[Set[int]]:
    # Apply Louvain method
    communities = lm.louvain_method(G, 1)
    
    # Plot the graph with communities
    if not benchmarking_mode:
        print("Modularity: ", utils.modularity(G, communities))

        utils.plot_graph_with_communities(
            G,
            communities,
            title=f"{dataset_name} with Communities Detected by our Implementation of the Louvain Method",
            label_edges=nx.is_weighted(G)
        )
        merged_G = utils.merge_communities(
            G,
            communities,
        )
        
        utils.plot_graph_with_communities(
            merged_G,
            communities=[{node} for node in merged_G.nodes()],
            label_edges=nx.is_weighted(G),
            title=f"{dataset_name} with Communities Detected by our Implementation of the Louvain Method",
        )
        
    return communities


if __name__ == "__main__":
    main()
