import time
import itertools
import csv
import math
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_ring_of_cliques(num_cliques, clique_size):
    G = nx.ring_of_cliques(num_cliques, clique_size)
    ground_truth = []
    node_offset = 0
    for cidx in range(num_cliques):
        community_nodes = set(range(node_offset, node_offset + clique_size))
        ground_truth.append(community_nodes)
        node_offset += clique_size
    return G, ground_truth

def generate_stochastic_block_model(sizes, probs):
    G = nx.stochastic_block_model(sizes, probs)
    ground_truth = []
    start = 0
    for block_size in sizes:
        end = start + block_size
        ground_truth.append(set(range(start, end)))
        start = end
    return G, ground_truth

def run_girvan_newman(G):
    import sys
    sys.path.append("../../Aidan/girvan_newman_implementation")
    import girvan_newman_implementation as gn
    return gn.main(G, False)

def run_girvan_newman_nx(G):
    return nx.community.girvan_newman(G)

def run_naive_greedy(G):
    from networkx.algorithms.community import naive_greedy_modularity_communities
    communities = naive_greedy_modularity_communities(G, resolution=1, weight=None)
    return list(communities)

def run_label_propagation(G):
    from networkx.algorithms.community import label_propagation_communities
    communities = label_propagation_communities(G)
    return list(communities)

def run_asyn_lpa(G):
    from networkx.algorithms.community import asyn_lpa_communities
    communities_iter = asyn_lpa_communities(G, weight=None, seed=None)
    return list(communities_iter)

def run_louvain_nx(G):
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight=None, resolution=1, threshold=1e-7, seed=None)
    return list(communities)

def run_louvain(G):
    import sys
    sys.path.append("../../Yang_Tan")
    import louvain_method_implementation as lm
    return lm.main(G)

def conductance(G, partition):
    community_list = list(partition)
    num_pairs = 0
    total_conductance = 0.0
    for C_i, C_j in itertools.combinations(community_list, 2):
        cond = nx.algorithms.cuts.conductance(G, C_i, C_j)
        total_conductance += cond
        num_pairs += 1
    return total_conductance / num_pairs if num_pairs > 0 else 0.0

def ground_truth_overlap(ground_truth, pred_partition):
    node_to_gt_label = {}
    gt_label = 0
    for gt_set in ground_truth:
        for n in gt_set:
            node_to_gt_label[n] = gt_label
        gt_label += 1
    node_to_pred_label = {}
    pred_label = 0
    for p_set in pred_partition:
        for n in p_set:
            node_to_pred_label[n] = pred_label
        pred_label += 1
    all_nodes = sorted(node_to_gt_label.keys())
    true_labels = [node_to_gt_label[n] for n in all_nodes]
    pred_labels = [node_to_pred_label[n] for n in all_nodes]
    return adjusted_rand_score(true_labels, pred_labels)

def run_single_scenario(scenario):
    scenario_type, scenario_name, scenario_args = scenario
    if scenario_type == "ring_of_cliques":
        num_cliques, clique_size = scenario_args
        G, ground_truth = generate_ring_of_cliques(num_cliques, clique_size)
    else:
        sizes, probs = scenario_args
        G, ground_truth = generate_stochastic_block_model(sizes, probs)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    gn_time = gn_mod = gn_cond = gn_gt = math.nan
    gn_partition = []
    ng_time = ng_mod = ng_cond = ng_gt = math.nan
    ng_partition = []
    lp_time = lp_mod = lp_cond = lp_gt = math.nan
    lp_partition = []
    alp_time = alp_mod = alp_cond = alp_gt = math.nan
    alp_partition = []
    lv_time = lv_mod = lv_cond = lv_gt = math.nan
    lv_partition = []
    try:
        t0 = time.time()
        gn_partition = run_girvan_newman(G)
        gn_time = time.time() - t0
        gn_mod = modularity(G, gn_partition)
        gn_cond = conductance(G, gn_partition)
        gn_gt = ground_truth_overlap(ground_truth, gn_partition)
    except Exception as e:
        print(f"[ERROR] Girvan–Newman failed in '{scenario_name}': {e}")
    try:
        t0 = time.time()
        ng_partition = run_naive_greedy(G)
        ng_time = time.time() - t0
        ng_mod = modularity(G, ng_partition)
        ng_cond = conductance(G, ng_partition)
        ng_gt = ground_truth_overlap(ground_truth, ng_partition)
    except Exception as e:
        print(f"[ERROR] Naive Greedy failed in '{scenario_name}': {e}")
    try:
        t0 = time.time()
        lp_partition = run_label_propagation(G)
        lp_time = time.time() - t0
        lp_mod = modularity(G, lp_partition)
        lp_cond = conductance(G, lp_partition)
        lp_gt = ground_truth_overlap(ground_truth, lp_partition)
    except Exception as e:
        print(f"[ERROR] Label Propagation failed in '{scenario_name}': {e}")
    try:
        t0 = time.time()
        alp_partition = run_asyn_lpa(G)
        alp_time = time.time() - t0
        alp_mod = modularity(G, alp_partition)
        alp_cond = conductance(G, alp_partition)
        alp_gt = ground_truth_overlap(ground_truth, alp_partition)
    except Exception as e:
        print(f"[ERROR] Asyn LPA failed in '{scenario_name}': {e}")
    try:
        t0 = time.time()
        lv_partition = run_louvain(G)
        lv_time = time.time() - t0
        lv_mod = modularity(G, lv_partition)
        lv_cond = conductance(G, lv_partition)
        lv_gt = ground_truth_overlap(ground_truth, lv_partition)
    except Exception as e:
        print(f"[ERROR] Louvain failed in '{scenario_name}': {e}")
    scenario_result = {
        "ScenarioType": scenario_type,
        "ScenarioName": scenario_name,
        "NumNodes": n_nodes,
        "NumEdges": n_edges,
        "GN_Communities": len(gn_partition),
        "GN_Time": gn_time,
        "GN_Modularity": gn_mod,
        "GN_Conductance": gn_cond,
        "GN_GroundTruth": gn_gt,
        "NG_Communities": len(ng_partition),
        "NG_Time": ng_time,
        "NG_Modularity": ng_mod,
        "NG_Conductance": ng_cond,
        "NG_GroundTruth": ng_gt,
        "LP_Communities": len(lp_partition),
        "LP_Time": lp_time,
        "LP_Modularity": lp_mod,
        "LP_Conductance": lp_cond,
        "LP_GroundTruth": lp_gt,
        "ALP_Communities": len(alp_partition),
        "ALP_Time": alp_time,
        "ALP_Modularity": alp_mod,
        "ALP_Conductance": alp_cond,
        "ALP_GroundTruth": alp_gt,
        "LV_Communities": len(lv_partition),
        "LV_Time": lv_time,
        "LV_Modularity": lv_mod,
        "LV_Conductance": lv_cond,
        "LV_GroundTruth": lv_gt,
    }
    return scenario_result

if __name__ == "__main__":
    roc_param_grid = []
    for num_cliques in range(2, 5):
        for clique_size in range(2, 5):
            roc_param_grid.append((num_cliques, clique_size))
    sbm_param_grid = []
    block_size_options = [
        [10, 10],
        [50, 50],
        [100, 100],
        [200, 200],
        [300, 300],
    ]
    p_in_values = [0.5]
    p_out_values = [0.5]
    for sizes in block_size_options:
        for p_in in p_in_values:
            for p_out in p_out_values:
                probs = [[p_in, p_out], [p_out, p_in]]
                sbm_param_grid.append((sizes, probs))
    scenarios = []
    for (num_cliques, clique_size) in roc_param_grid:
        name = f"ROC_{num_cliques}x{clique_size}"
        scenarios.append(("ring_of_cliques", name, (num_cliques, clique_size)))
    for (sizes, probs) in sbm_param_grid:
        total_nodes = sum(sizes)
        name = f"SBM_{len(sizes)}blocks_{total_nodes}nodes_pin{probs[0][0]:.3f}_pout{probs[0][1]:.3f}"
        scenarios.append(("sbm", name, (sizes, probs)))
    total_scenarios = len(scenarios)
    print(f"Starting parallel benchmark on {total_scenarios} scenarios...\n")
    fieldnames = [
        "ScenarioType", "ScenarioName", "NumNodes", "NumEdges",
        "GN_Communities", "GN_Time", "GN_Modularity", "GN_Conductance", "GN_GroundTruth",
        "NG_Communities", "NG_Time", "NG_Modularity", "NG_Conductance", "NG_GroundTruth",
        "LP_Communities", "LP_Time", "LP_Modularity", "LP_Conductance", "LP_GroundTruth",
        "ALP_Communities", "ALP_Time", "ALP_Modularity", "ALP_Conductance", "ALP_GroundTruth",
        "LV_Communities", "LV_Time", "LV_Modularity", "LV_Conductance", "LV_GroundTruth",
    ]
    max_workers = 15
    with open("benchmark_log_batch2.csv", "w", newline="", buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_scenario = {
                executor.submit(run_single_scenario, scenario): scenario
                for scenario in scenarios
            }
            for idx, future in enumerate(as_completed(future_to_scenario), start=1):
                scenario = future_to_scenario[future]
                try:
                    row_result = future.result()
                except Exception as e:
                    print(f"[FATAL ERROR] in {scenario}")
                    print(e)
                    continue
                writer.writerow(row_result)
                s_type, s_name, _ = scenario
                print(f"[{idx}/{total_scenarios}] Completed: {s_type} - {s_name}")