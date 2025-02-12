import time
import itertools
import csv
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score
import math  # for math.nan if desired


def run_louvain(G):
    """
    Uses your custom Louvain method from 'louvain_method_implementation'.
    """
    import sys
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm
    return lm.main(G)

G = nx.Graph()
    # Add nodes with attributes
G.add_node(1, label="A")
G.add_node(2, label="B")
G.add_node(3, label="C")
G.add_node(4, label="D")
G.add_node(5, label="E")
G.add_node(6, label="F")
G.add_node(7, label="G")
G.add_node(8, label="H")

# # Add weighted edges
G.add_edges_from([
    (1, 2, {'weight': 1}),
     (1, 3, {'weight': 2}),
     (2, 3, {'weight': 1}),
     (3, 4, {'weight': 1}),
     (4, 5, {'weight': 2}),
     (5, 6, {'weight': 1}),
     (6, 7, {'weight': 2}),
     (5, 7, {'weight': 1}),
     (6, 8, {'weight': 2}),
     (7, 8, {'weight': 1}),
 ])
run_louvain(G)