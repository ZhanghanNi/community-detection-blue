import sys
sys.path.append("../../Util")
import utils
import random
import networkx as nx
import math
import copy

"""
Code written by Jake Jasmer, February 2025

This code can be used to run the Basic Variable Neighborhood Search algorithm for community detection
as described in "Variable Neighborhood Search Approach to Community Detection Problem" by 
Jovanovic et al. It starts with a randomly generated set of communities and then iterates over the
basic steps. The basic steps involve: 1) shaking: changing the solution by a few nodes to get out
of local optimums, 2) local search: in the shaken solution, search around for the best local solution,
and 3) comparison: compare the previously best solution with the new local optimum.

This algorithm is randomized in the starting solution and in the shaking step, so it is likely that
running it multiple times on the same graph will not yield the same solution. Consider running it a few
times and taking the average. 
"""

def main(G: nx.Graph, kmax: int = 3, dataset_name: str = "Graph"):
    """
    Main function to be called by main.py that runs the implementation of BVNS on
    a specified data set
    """
    immutable_G = G
    G_to_pass = G.copy()

    bvns_communities = bvns(G_to_pass, kmax)
    print("Modularity: ", utils.modularity(immutable_G, bvns_communities))
    utils.plot_graph_with_communities(
        immutable_G,
        bvns_communities,
        label_edges=True,
        title=f"{dataset_name} Communities Detected by our Implementation of Basic Variable Neighborhood Search",
    )

    merged_G = utils.merge_communities(immutable_G, bvns_communities)

    utils.plot_graph_with_communities(
        merged_G,
        communities=[{node} for node in merged_G.nodes()],
        label_edges=True,
        title=f"{dataset_name} Communities Detected by our Implementation of Basic Variable Neighborhood Search",
    )

"""
This function runs bvns starting with a differing number of communities form 2-max_communities and
returns the result with the best modularity. The starting communities are randomized
"""
def bvns(graph, kmax=3, max_communities=None, stop_criterion=100):
    if max_communities is None:
        max_communities = int(len(graph.nodes) ** 0.5) + 1
    best_modularity = -math.inf
    best_communities = []
    for num_communities in range(2, max_communities):
        new_communities = bvns_with_communities_specified(graph, num_communities, kmax, stop_criterion)
        new_modularity = utils.modularity(graph, new_communities)
        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_communities = copy.deepcopy(new_communities)
    return utils.remove_empty_communities(best_communities)


"""
This function runs bvns on a set number of communities
"""
def bvns_with_communities_specified(graph, num_communities=None, kmax=3, stop_criterion=100):
    if num_communities is None:
        num_communities = int(len(graph.nodes) ** 0.5) + 1
    # Start with a random solution
    random_groups = random_grouping(graph, num_communities)
    best_modulartiy = utils.modularity(graph, random_groups)
    current_best_communities = random_groups
    
    
    # Assuming stop_criterion is an int
    for _ in range(stop_criterion):
        # Shake
        shaken_communities = shake(graph, current_best_communities, kmax)
        # Find a local optimum
        local_best = local_search(graph, shaken_communities, kmax)
        # Compare to current best to best and choose the max
        local_modularity = utils.modularity(graph, local_best)
        if local_modularity > best_modulartiy:
            current_best_communities = local_best
            best_modulartiy = local_modularity
    return utils.remove_empty_communities(current_best_communities)

"""
This function creates a random partition of nodes into a given number of communities.
It is used at the start of bvns
"""
def random_grouping(graph, num_communities):
    nodes = list(graph.nodes())
    # Generate a random number for the number of starting communities, can change this if need be
    random.shuffle(nodes) # Shuffle the list
    # Split into groups
    groups = [[] for _ in range(num_communities)]
    for i, node in enumerate(nodes):
        groups[i % num_communities].append(node)  # Distribute nodes cyclically

    return groups 

"""
This method romdomly generates new sets of communities where k nodes are moved from k=1-kmax.
It then picks the best one to use for local search
"""
def shake(graph, communities, kmax):
    best_modularity = -math.inf
    best_communities = communities
    for k in range(1, kmax+1):
        new_communities = change_communities(graph, communities, k)
        new_modularity = utils.modularity(graph, new_communities)
        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_communities = new_communities
    return best_communities

"""
This function is used by the shake function to randomly change k nodes in the set of communities
"""
def change_communities(graph, communities, k):
    random_nodes = []
    for k in range(k):
        communities_out = copy.deepcopy(communities)
        # Find k random nodes
        random_nodes.append(random.choice(list(graph.nodes())))
    # find a change of community for each node
    for node in random_nodes:
        # Find its community
        community_num = -1
        for i in range(len(communities_out)):
            if node in communities_out[i]:
                community_num = i
                break
        communities_out[community_num].remove(node)
        # Choose a different one to move it to (skipping community_num so it doesn't go back to the same community)
        # This would be len(communities_out) -2, but we want to include a new empty community as a possibility
        new_community = random.randint(0, len(communities_out) -1)
        # Avoid choosing the original community
        if new_community >= community_num:
            new_community += 1
        if new_community >= len(communities):
            # If the chosen number isn't a community, create a new one
            communities_out.append([node])
        else:
            communities_out[new_community].append(node)
    return communities_out


"""
This function is used to run a local search on a set of communities in a graph.
It looks at the given communities and all permutations of one node changing and picks the best
modularity. It is meant to be a way of implementing gradient descent
"""
def local_search(graph, communities, kmax=1):
    best_modularity = utils.modularity(graph, communities)
    best_communities = communities
    for i in range(len(communities)):
        community = communities[i]
        for node in community:
            # move it to each other community and check if modularity is better
            for j in range(len(communities)):
                if j != i:
                    new_communities = copy.deepcopy(communities)
                    new_communities[i].remove(node)
                    new_communities[j].append(node)
                    new_modularity = utils.modularity(graph, new_communities)
                    if new_modularity > best_modularity:
                        best_modularity = new_modularity
                        best_communities = copy.deepcopy(new_communities)
            # Also try moving node to a new community
            new_communities = copy.deepcopy(communities)
            new_communities[i].remove(node)
            # Add a new community with just node
            new_communities.append([node])
            new_modularity = utils.modularity(graph, new_communities)
            if new_modularity > best_modularity:
                        best_modularity = new_modularity
                        best_communities = copy.deepcopy(new_communities)
    return best_communities
