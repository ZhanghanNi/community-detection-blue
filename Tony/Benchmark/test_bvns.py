import time
import itertools
import csv
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score
import math  # for math.nan if desired


def run_bvns(G):
    import sys
    sys.path.append("../../Jake")
    import louvain_method_implementation as lm
    return lm.main(G)


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
run_louvain(G)