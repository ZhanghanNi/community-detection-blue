import csv
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Set, Dict, Union
from networkx.algorithms.community import girvan_newman, louvain_communities
from networkx.algorithms.community.quality import modularity
import itertools

def prepare_network(input_file: str, output_file: str, filters: Dict[str, float]) -> None:
    df = pd.read_csv(input_file, sep=r'\s+')
    
    mask = pd.Series(True, index=df.index)
    for col, threshold in filters.items():
        mask &= (df[col] >= threshold)
    
    df_filtered = df[mask].copy()
    df_filtered['protein1'] = df_filtered['protein1'].str.replace(r'^4932\.', '', regex=True)
    df_filtered['protein2'] = df_filtered['protein2'].str.replace(r'^4932\.', '', regex=True)
    
    df_filtered[['protein1', 'protein2']].to_csv(output_file, header=False, index=False, sep=' ')

def detect_communities(G: nx.Graph, method: str = 'louvain') -> List[Set]:
    if method == 'louvain':
        return louvain_communities(G, weight=None)
    
    elif method == 'girvan_newman':
        comp = girvan_newman(G)
        # take communities while they improve modularity
        best_communities = next(comp)
        best_modularity = modularity(G, best_communities)
        
        for communities in itertools.takewhile(
            lambda c: len(c) <= int(np.sqrt(G.number_of_nodes())), 
            comp
        ):
            current_modularity = modularity(G, communities)
            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_communities = communities
        
        return list(best_communities)

def evaluate_clustering(pred_clusters: List[Set], ref_clusters: List[Set], 
                       overlap_threshold: float = 0.3) -> Dict[str, float]:
    def calc_overlap_metrics(clusters1: List[Set], clusters2: List[Set], 
                           threshold: float) -> float:
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
            return sum((len(c1 & c2)**2) / (len(c1) * len(c2))
                      for c1 in c1s for c2 in c2s
                      if len(c1) > 0 and len(c2) > 0)

        sum1 = sum_intersections(p_clusters, r_clusters)
        sum2 = sum_intersections(r_clusters, p_clusters)
        denom = len(p_clusters) * len(r_clusters)
        
        return np.sqrt((sum1 * sum2) / (denom**2))

    recall = calc_overlap_metrics(ref_clusters, pred_clusters, overlap_threshold)
    precision = calc_overlap_metrics(pred_clusters, ref_clusters, overlap_threshold)
    f_measure = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    
    ppv = calc_value_metrics(pred_clusters, ref_clusters)
    sn = calc_value_metrics(ref_clusters, pred_clusters)
    acc = np.sqrt(ppv * sn)
    
    return {
        "Recall": recall,
        "Precision": precision,
        "F-measure": f_measure,
        "PPV": ppv,
        "SN": sn,
        "ACC": acc,
        "SEP": calc_separation(pred_clusters, ref_clusters),
        "NumClusters": len(pred_clusters)
    }

def load_data(file_path: str, data_type: str = 'network') -> Union[nx.Graph, List[Set]]:
    if data_type == 'network':
        return nx.read_edgelist(file_path, comments='#', nodetype=str)
    
    elif data_type == 'clusters':
        clusters = []
        with open(file_path) as f:
            clusters = [set(line.strip().split()) for line in f]
        return clusters
        
