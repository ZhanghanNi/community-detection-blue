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
    return [set(lst) for lst in bvns.bvns_with_communities_specified(G)]

def run_louvain_nx(G):
    from networkx.algorithms.community import louvain_communities
    c = louvain_communities(G, weight=None, resolution=1, threshold=1e-7, seed=None)
    return list(c)

def run_girvan_newman_nx(G):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return list(gn.find_network_x_communities_with_highest_modularity(G,G.number_of_nodes()))


def ring_of_cliques_density(num_cliques, clique_size):
    n = num_cliques * clique_size
    intra_edges = num_cliques * (clique_size * (clique_size - 1) // 2)
    ring_edges = num_cliques if num_cliques >= 2 else 0
    e = intra_edges + ring_edges
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
        cleaned_gt = []
        for s in ground_truth:
            s2 = s - set(iso)
            if s2:
                cleaned_gt.append(s2)
        ground_truth = cleaned_gt
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



import tracemalloc

def measure_performance(algorithm, graph, kmax=3):
    tracemalloc.start()
    start_time = time.time()
    

    partition = algorithm(graph)
    
    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_used = peak
    return partition, elapsed_time, memory_used


def run_single_scenario(scenario):
    density_label, stype, sname, args = scenario

    if stype == "ring_of_cliques":
        G, gt = generate_ring_of_cliques(*args)
    else:
        G, gt = generate_stochastic_block_model(args[0], args[1], args[2])

    n = G.number_of_nodes()
    e = G.number_of_edges()

    try:
        gnp, gnt, gnm = measure_performance(run_girvan_newman, G)
    except Exception as exc:
        gnp, gnt, gnm = [], math.nan, math.nan
    try:
        lpp, lpt, lpm = measure_performance(run_label_propagation, G)
    except Exception as exc:
        lpp, lpt, lpm = [], math.nan, math.nan
    try:
        alpp, alpt, alpm = measure_performance(run_asyn_lpa, G)
    except Exception as exc:
        alpp, alpt, alpm = [], math.nan, math.nan
    try:
        lvp, lvt, lvm = measure_performance(run_louvain, G)
    except Exception as exc:
        lvp, lvt, lvm = [], math.nan, math.nan
    try:
        bvpp, bvpt, bvpm = measure_performance(run_bvns, G)
    except Exception as exc:
        bvpp, bvpt, bvpm = [], math.nan, math.nan

    try:
        gnxp, gnxt, gnxm = measure_performance(run_girvan_newman_nx, G)
    except Exception as exc:
        gnxp, gnxt, gnxm = [], math.nan, math.nan
    try:
        lnxp, lnxt, lnxm = measure_performance(run_louvain_nx, G)
    except Exception as exc:
        lnxp, lnxt, lnxm = [], math.nan, math.nan

    def compute_metrics(part):
        if part:
            return (
                modularity(G, part),
                conductance(G, part),
                ground_truth_overlap(gt, part),
            )
        return (math.nan, math.nan, math.nan)

    gm, gc, gari = compute_metrics(gnp)
    lpm_, lpc, lpa = compute_metrics(lpp)
    alpm_, alpc, alpa = compute_metrics(alpp)
    lvm_, lvc, lva = compute_metrics(lvp)
    bvnm, bvnc, bvna = compute_metrics(bvpp)
    gnxm_, gnxcond, gnxari = compute_metrics(gnxp)
    lnxm_, lnxcond, lnxari = compute_metrics(lnxp)

    return {
        "ScenarioDensity": density_label,
        "ScenarioType": stype,
        "ScenarioName": sname,
        "NumNodes": n,
        "NumEdges": e,

        # Girvan-Newman (ours)
        "GN_Communities": len(gnp),
        "GN_Time": gnt,
        "GN_Memory": gnm,
        "GN_Modularity": gm,
        "GN_Conductance": gc,
        "GN_GroundTruth": gari,

        # Label Propagation
        "LP_Communities": len(lpp),
        "LP_Time": lpt,
        "LP_Memory": lpm,
        "LP_Modularity": lpm_,
        "LP_Conductance": lpc,
        "LP_GroundTruth": lpa,

        # Asyn LPA
        "ALP_Communities": len(alpp),
        "ALP_Time": alpt,
        "ALP_Memory": alpm,
        "ALP_Modularity": alpm_,
        "ALP_Conductance": alpc,
        "ALP_GroundTruth": alpa,

        # Louvain (ours)
        "LV_Communities": len(lvp),
        "LV_Time": lvt,
        "LV_Memory": lvm,
        "LV_Modularity": lvm_,
        "LV_Conductance": lvc,
        "LV_GroundTruth": lva,

        # BVNS
        "BVNS_Communities": len(bvpp),
        "BVNS_Time": bvpt,
        "BVNS_Memory": bvpm,
        "BVNS_Modularity": bvnm,
        "BVNS_Conductance": bvnc,
        "BVNS_GroundTruth": bvna,

        # Nx Girvan-Newman
        "GNX_Communities": len(gnxp),
        "GNX_Time": gnxt,
        "GNX_Memory": gnxm,
        "GNX_Modularity": gnxm_,
        "GNX_Conductance": gnxcond,
        "GNX_GroundTruth": gnxari,

        # Nx Louvain
        "LNX_Communities": len(lnxp),
        "LNX_Time": lnxt,
        "LNX_Memory": lnxm,
        "LNX_Modularity": lnxm_,
        "LNX_Conductance": lnxcond,
        "LNX_GroundTruth": lnxari,
    }


def random_ring_scenarios(target_density, count):
    scenarios = []
    tries = 0
    while len(scenarios) < count and tries < 50000:
        c = random.randint(1, 50)
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

if __name__ == "__main__":
    random.seed(0)

    # scenario generation
    sparse_ring = random_ring_scenarios(0.1, 66)
    interm_ring = random_ring_scenarios(0.3, 66)
    dense_ring = random_ring_scenarios(0.5, 66)
    sparse_sbm = random_sbm_scenarios(0.1, 66)
    interm_sbm = random_sbm_scenarios(0.5, 66)
    dense_sbm = random_sbm_scenarios(0.8, 66)
    
    scenarios_sparse = sparse_ring + sparse_sbm
    scenarios_interm = interm_ring + interm_sbm
    scenarios_dense = dense_ring + dense_sbm

    all_scenarios = (
        [("SPARSE", sc[0], sc[1], sc[2]) for sc in scenarios_sparse] +
        [("INTERMEDIATE", sc[0], sc[1], sc[2]) for sc in scenarios_interm] +
        [("DENSE", sc[0], sc[1], sc[2]) for sc in scenarios_dense]
    )

    fnames = [
        "ScenarioDensity","ScenarioType","ScenarioName","NumNodes","NumEdges",

        "GN_Communities","GN_Time","GN_Memory","GN_Modularity","GN_Conductance","GN_GroundTruth",
        "LP_Communities","LP_Time","LP_Memory","LP_Modularity","LP_Conductance","LP_GroundTruth",
        "ALP_Communities","ALP_Time","ALP_Memory","ALP_Modularity","ALP_Conductance","ALP_GroundTruth",
        "LV_Communities","LV_Time","LV_Memory","LV_Modularity","LV_Conductance","LV_GroundTruth",
        "BVNS_Communities","BVNS_Time","BVNS_Memory","BVNS_Modularity","BVNS_Conductance","BVNS_GroundTruth",

        "GNX_Communities","GNX_Time","GNX_Memory","GNX_Modularity","GNX_Conductance","GNX_GroundTruth",
        "LNX_Communities","LNX_Time","LNX_Memory","LNX_Modularity","LNX_Conductance","LNX_GroundTruth",
    ]

    master_results = []
    with open("benchmark_log_FULL_corrected.csv", "w", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fnames)
        writer.writeheader()

        # parallel execution of all scenarios
        with ProcessPoolExecutor(max_workers=50) as executor:
            future_to_scenario = {
                executor.submit(run_single_scenario, scn): scn
                for scn in all_scenarios
            }
            idx = 1
            for fut in as_completed(future_to_scenario):
                scn = future_to_scenario[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"[ERROR] {scn}: {e}")
                    continue
                writer.writerow(row)
                master_results.append(row)
                print(f"[{idx}/{len(all_scenarios)}] {row['ScenarioName']} - {row['ScenarioDensity']} done")
                idx += 1

  