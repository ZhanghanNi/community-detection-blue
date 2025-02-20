import time
import itertools
import csv
import math
import os
import psutil
import random
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

def run_bvns(G):
    import sys
    sys.path.append("../../Jake/bvns")
    import bvns_implementation as bvns
    return [set(lst) for lst in bvns.bvns(G)]

def ring_of_cliques_density(num_cliques, clique_size):
    n = num_cliques * clique_size
    e = num_cliques * (clique_size * (clique_size - 1) // 2) + num_cliques
    return (2 * e) / (n * (n - 1)) if n > 1 else 0.0

def generate_ring_of_cliques(num_cliques, clique_size):
    G = nx.ring_of_cliques(num_cliques, clique_size)
    ground_truth = []
    node_offset = 0
    for _ in range(num_cliques):
        community_nodes = set(range(node_offset, node_offset + clique_size))
        ground_truth.append(community_nodes)
        node_offset += clique_size
    return G, ground_truth

def remove_isolates(G, ground_truth):
    iso = [n for n, deg in G.degree() if deg == 0]
    if iso:
        G.remove_nodes_from(iso)
        clean_gt = []
        for s in ground_truth:
            s2 = s - set(iso)
            if s2:
                clean_gt.append(s2)
        ground_truth = clean_gt
    return G, ground_truth

def generate_stochastic_block_model(sizes, p_in, p_out):
    G = nx.Graph()
    ground_truth = []
    start = 0
    for sz in sizes:
        end = start + sz
        ground_truth.append(set(range(start, end)))
        start = end
    n_blocks = len(sizes)
    probs = [[p_out] * n_blocks for _ in range(n_blocks)]
    for i in range(n_blocks):
        probs[i][i] = p_in
    G = nx.stochastic_block_model(sizes, probs)
    G, ground_truth = remove_isolates(G, ground_truth)
    return G, ground_truth

def approximate_sbm_density(sizes, p_in, p_out):
    n = sum(sizes)
    k = len(sizes)
    e_in = 0
    for sz in sizes:
        e_in += p_in * (sz * (sz - 1) / 2)
    e_out = 0
    for i in range(k):
        for j in range(i + 1, k):
            e_out += p_out * (sizes[i] * sizes[j])
    e_approx = e_in + e_out
    return (2 * e_approx) / (n * (n - 1)) if n > 1 else 0.0

def run_girvan_newman(G):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return gn.main(G, False)

def run_label_propagation(G):
    from networkx.algorithms.community import label_propagation_communities
    return list(label_propagation_communities(G))

def run_asyn_lpa(G):
    from networkx.algorithms.community import asyn_lpa_communities
    return list(asyn_lpa_communities(G, weight=None, seed=None))

def run_louvain(G):
    import sys
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm
    return lm.main(G)

def conductance(G, partition):
    c = list(partition)
    s = 0
    t = 0
    for C_i, C_j in itertools.combinations(c, 2):
        s += nx.algorithms.cuts.conductance(G, C_i, C_j)
        t += 1
    return s / t if t else 0

def ground_truth_overlap(ground_truth, pred_partition):
    label_map = {}
    gidx = 0
    for gt_set in ground_truth:
        for n in gt_set:
            label_map[n] = gidx
        gidx += 1
    pred_map = {}
    pidx = 0
    for s in pred_partition:
        for n in s:
            pred_map[n] = pidx
        pidx += 1
    common_nodes = sorted(label_map.keys() & pred_map.keys())
    return adjusted_rand_score(
        [label_map[x] for x in common_nodes],
        [pred_map[x] for x in common_nodes]
    )

def measure_method(method_func, G):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    t0 = time.time()
    partition = method_func(G)
    elapsed = time.time() - t0
    mem_after = process.memory_info().rss
    mem_used = mem_after - mem_before
    return partition, elapsed, mem_used

def run_single_scenario(scenario):
    density_label, stype, sname, args = scenario
    if stype == "ring_of_cliques":
        G, gt = generate_ring_of_cliques(*args)
    else:
        G, gt = generate_stochastic_block_model(args[0], args[1], args[2])

    n = G.number_of_nodes()
    e = G.number_of_edges()

    gnp, gnt, gnm = [], math.nan, math.nan
    lpp, lpt, lpm = [], math.nan, math.nan
    alpp, alpt, alpm = [], math.nan, math.nan
    lvp, lvt, lvm = [], math.nan, math.nan
    bvpp, bvpt, bvpm = [], math.nan, math.nan

    try:
        gnp, gnt, gnm = measure_method(run_girvan_newman, G)
    except:
        pass
    try:
        lpp, lpt, lpm = measure_method(run_label_propagation, G)
    except:
        pass
    try:
        alpp, alpt, alpm = measure_method(run_asyn_lpa, G)
    except:
        pass
    try:
        lvp, lvt, lvm = measure_method(run_louvain, G)
    except:
        pass
    # BVNS
    try:
        bvpp, bvpt, bvpm = measure_method(run_bvns, G)
    except:
        pass

    gm = condv = gtari = math.nan
    if gnp:
        gm = modularity(G, gnp)
        condv = conductance(G, gnp)
        gtari = ground_truth_overlap(gt, gnp)

    lpmod = lpcond = lparti = math.nan
    if lpp:
        lpmod = modularity(G, lpp)
        lpcond = conductance(G, lpp)
        lparti = ground_truth_overlap(gt, lpp)

    alpmod = alpcond = alparti = math.nan
    if alpp:
        alpmod = modularity(G, alpp)
        alpcond = conductance(G, alpp)
        alparti = ground_truth_overlap(gt, alpp)

    lvmod = lvcond = lvarti = math.nan
    if lvp:
        lvmod = modularity(G, lvp)
        lvcond = conductance(G, lvp)
        lvarti = ground_truth_overlap(gt, lvp)

    bvmod = bvcond = bvarti = math.nan
    if bvpp:
        bvmod = modularity(G, bvpp)
        bvcond = conductance(G, bvpp)
        bvarti = ground_truth_overlap(gt, bvpp)

    return {
        "ScenarioDensity": density_label,
        "ScenarioType": stype,
        "ScenarioName": sname,
        "NumNodes": n,
        "NumEdges": e,
        "GN_Communities": len(gnp),
        "GN_Time": gnt,
        "GN_Memory": gnm,
        "GN_Modularity": gm,
        "GN_Conductance": condv,
        "GN_GroundTruth": gtari,
        "LP_Communities": len(lpp),
        "LP_Time": lpt,
        "LP_Memory": lpm,
        "LP_Modularity": lpmod,
        "LP_Conductance": lpcond,
        "LP_GroundTruth": lparti,
        "ALP_Communities": len(alpp),
        "ALP_Time": alpt,
        "ALP_Memory": alpm,
        "ALP_Modularity": alpmod,
        "ALP_Conductance": alpcond,
        "ALP_GroundTruth": alparti,
        "LV_Communities": len(lvp),
        "LV_Time": lvt,
        "LV_Memory": lvm,
        "LV_Modularity": lvmod,
        "LV_Conductance": lvcond,
        "LV_GroundTruth": lvarti,
        "BVNS_Communities": len(bvpp),
        "BVNS_Time": bvpt,
        "BVNS_Memory": bvpm,
        "BVNS_Modularity": bvmod,
        "BVNS_Conductance": bvcond,
        "BVNS_GroundTruth": bvarti
    }

# def random_ring_scenarios(target_density, count):
#     scenarios = []
#     tries = 0
#     while len(scenarios) < count and tries < 50000:
#         c = random.randint(2,50)
#         k = random.randint(2,50)
#         d = ring_of_cliques_density(c,k)
#         if abs(d - target_density) < 0.02:
#             name = f"RING_{int(target_density*100)}_{len(scenarios)+1}"
#             scenarios.append(("ring_of_cliques", name, (c,k)))
#         tries += 1
#     return scenarios

# def random_sbm_scenarios(target_density, count):
#     scenarios = []
#     tries = 0
#     while len(scenarios) < count and tries < 50000:
#         blocks = random.randint(2,5)
#         sizes = [random.randint(5,40) for _ in range(blocks)]
#         p_in = round(random.uniform(0.01,0.20),3)
#         p_out = round(random.uniform(0.001,0.15),3)
#         d = approximate_sbm_density(sizes, p_in, p_out)
#         if abs(d - target_density) < 0.02:
#             name = f"SBM_{int(target_density*100)}_{len(scenarios)+1}"
#             scenarios.append(("sbm", name, (sizes, p_in, p_out)))
#         tries += 1
#     return scenarios

def ring_of_cliques_density(num_cliques, clique_size):
    n = num_cliques * clique_size
    # Correct edge calculation: no ring edges if num_cliques == 1
    intra_edges = num_cliques * (clique_size * (clique_size - 1) // 2)
    ring_edges = num_cliques if num_cliques >= 2 else 0
    e = intra_edges + ring_edges
    return (2 * e) / (n * (n - 1)) if n > 1 else 0.0

def random_ring_scenarios(target_density, count):
    scenarios = []
    tries = 0
    while len(scenarios) < count and tries < 50000:
        # Allow c=1 to generate single-clique (high-density) scenarios
        c = random.randint(1, 50)  # Changed from (2,50)
        k = random.randint(2, 50)
        d = ring_of_cliques_density(c, k)
        if abs(d - target_density) < 0.02:
            name = f"RING_{int(target_density*100)}_{len(scenarios)+1}"
            scenarios.append(("ring_of_cliques", name, (c, k)))
        tries += 1
    return scenarios

def random_sbm_scenarios(target_density, count):
    scenarios = []
    tries = 0
    while len(scenarios) < count and tries < 50000:
        blocks = random.randint(2, 5)
        sizes = [random.randint(5, 40) for _ in range(blocks)]
        # Adjust probability ranges based on target density
        if target_density >= 0.5:
            p_in = round(random.uniform(0.5, 1.0), 3)
            p_out = round(random.uniform(0.3, 1.0), 3)
        else:
            p_in = round(random.uniform(0.01, 0.20), 3)
            p_out = round(random.uniform(0.001, 0.15), 3)
        d = approximate_sbm_density(sizes, p_in, p_out)
        if abs(d - target_density) < 0.02:
            name = f"SBM_{int(target_density*100)}_{len(scenarios)+1}"
            scenarios.append(("sbm", name, (sizes, p_in, p_out)))
        tries += 1
    return scenarios

    
sparse_ring = random_ring_scenarios(0.1, 50)
interm_ring = random_ring_scenarios(0.3, 50)
dense_ring = random_ring_scenarios(0.5, 50)
sparse_sbm = random_sbm_scenarios(0.1, 50)
interm_sbm = random_sbm_scenarios(0.5, 50)
dense_sbm = random_sbm_scenarios(0.8, 50)
print(len(sparse_ring))
print(len(interm_ring))
print(len(dense_ring))

print(len(sparse_sbm))
print(len(interm_sbm))
print(len(dense_sbm))