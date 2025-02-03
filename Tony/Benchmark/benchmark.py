import time
import itertools
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from networkx.algorithms.community.quality import modularity

def generate_ring_of_cliques(num_cliques, clique_size):
    G = nx.ring_of_cliques(num_cliques, clique_size)
    ground_truth = []
    node_offset = 0
    for cidx in range(num_cliques):
        community_nodes = set(range(node_offset, node_offset + clique_size))
        ground_truth.append(community_nodes)
        node_offset += clique_size
    return G, ground_truth

def generate_stochastic_block_model(sizes, probs, seed=0):
    G = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False, selfloops=False)
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
    return gn.main(G,False)


def run_louvain(G):
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight=None, resolution=1, threshold=1e-7, seed=None)
    return list(communities)


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

    # ensure same node ordering:
    all_nodes = sorted(node_to_gt_label.keys())
    true_labels = [node_to_gt_label[n] for n in all_nodes]
    pred_labels = [node_to_pred_label[n] for n in all_nodes]

    return adjusted_rand_score(true_labels, pred_labels)
    
if __name__ == "__main__":

    scenarios = []
    roc_params = [
        (4, 5),
        (8, 5),
        (4, 10),
    ]
    for (num_cliques, clique_size) in roc_params:
        name = f"RingC_{num_cliques}x{clique_size}"
        scenarios.append(("ring_of_cliques", name, (num_cliques, clique_size)))



    sbm_params = [
        ([50, 50], [[0.10, 0.01], [0.01, 0.10]]),
        ([100, 100], [[0.05, 0.005], [0.005, 0.05]]),
    ]
    for (sizes, probs) in sbm_params:
        name = f"SBM_{len(sizes)}blocks_{sum(sizes)}nodes"
        scenarios.append(("sbm", name, (sizes, probs)))

    results = []

    for scenario_type, scenario_name, scenario_args in scenarios:
        # ground truth
        if scenario_type == "ring_of_cliques":
            num_cliques, clique_size = scenario_args
            G, ground_truth = generate_ring_of_cliques(num_cliques, clique_size)
        else:
            sizes, probs = scenario_args
            G, ground_truth = generate_stochastic_block_model(sizes, probs, seed=42)
            print(G)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Girvan–Newman
        t0 = time.time()
        gn_partition = run_girvan_newman(G)
        print(gn_partition)
        t1 = time.time()
        gn_time = t1 - t0
        gn_mod = modularity(G, gn_partition)
        gn_cond = conductance(G, gn_partition)
        gn_gt = ground_truth_overlap(ground_truth, gn_partition)


        # Louvain
        t2 = time.time()
        lv_partition = run_louvain(G)
        t3 = time.time()
        lv_time = t3 - t2
        lv_mod = modularity(G, lv_partition)
        lv_cond = conductance(G, lv_partition)
        lv_gt = ground_truth_overlap(ground_truth, lv_partition)

      
        results.append({
            "Scenario": scenario_name,
            "NumNodes": n_nodes,
            "NumEdges": n_edges,

            "GN_Communities": len(gn_partition),
            "GN_Time": gn_time,
            "GN_Modularity": gn_mod,
            "GN_Conductance": gn_cond,
            "GN_GroundTruth": gn_gt,

            "LV_Communities": len(lv_partition),
            "LV_Time": lv_time,
            "LV_Modularity": lv_mod,
            "LV_Conductance": lv_cond,
            "LV_GroundTruth": lv_gt,
        })

    print("\n===== BENCHMARK RESULTS =====")
    for row in results:
        print(f"\nScenario: {row['Scenario']} "
              f"({row['NumNodes']} nodes, {row['NumEdges']} edges)")
        print(f"  Girvan–Newman: communities={row['GN_Communities']}  "
              f"time={row['GN_Time']:.4f}s  "
              f"mod={row['GN_Modularity']:.3f}  cond={row['GN_Conductance']:.3f}  gt={row['GN_GroundTruth']:.3f}")
        print(f"  Louvain:       communities={row['LV_Communities']}  "
              f"time={row['LV_Time']:.4f}s  "
              f"mod={row['LV_Modularity']:.3f}  cond={row['LV_Conductance']:.3f}  gt={row['LV_GroundTruth']:.3f}")