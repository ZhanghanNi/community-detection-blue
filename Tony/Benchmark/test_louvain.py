import time
import itertools
import csv
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score
import math  # for math.nan if desired


def run_louvain(G):
    import sys
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm
    return lm.main(G)

def run_girvan_newman_nx(G):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return list(gn.find_network_x_communities_with_highest_modularity(G,G.number_of_nodes()))


G = nx.ring_of_cliques(2, 2)


# # Add weighted edges
# G.add_edges_from([
#     (1, 2, {'weight': 1}),
#      (1, 3, {'weight': 2}),
#      (2, 3, {'weight': 1}),
#      (3, 4, {'weight': 1}),
#      (4, 5, {'weight': 2}),
#      (5, 6, {'weight': 1}),
#      (6, 7, {'weight': 2}),
#      (5, 7, {'weight': 1}),
#      (6, 8, {'weight': 2}),
#      (7, 8, {'weight': 1}),
#  ])
print(run_girvan_newman_nx(G))