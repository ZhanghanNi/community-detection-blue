import networkx as nx
import sys
sys.path.append("../Util/")
sys.path.append("../Aidan/girvan_newman_implementation")
import utils
import girvan_newman as gn
import girvan_newman as gni
from collections import defaultdict


def find_football_truth_values():
    # Load the GML file
    # G = nx.read_gml("./football/football.gml")
    G = nx.read_gml("Jake/football/football.gml")

    # Group nodes by their "value" (community)
    community_dict = defaultdict(set)

    for node, data in G.nodes(data=True):
        community_dict[data["value"]].add(node)

    # Convert to a list of sets
    ground_truth_communities = list(community_dict.values())

    return ground_truth_communities
    # out = ""
    # # Return the communities
    # for i, community in enumerate(ground_truth_communities, 1):
    #     out += f"Community {i}: {community}" + "\n"
    # return out


def find_girvan_newman_communities():
    G = nx.read_gml("./football/football.gml")
    solution = gni.main(G, True, "football")
    print("--------------------------------------\nGirvan Newman Answers\n--------------------------------------")
    out = ""
    # Return the communities
    for i, community in enumerate(solution, 1):
        out += f"Community {i}: {community}" + "\n"
    return out

def main():
    print(find_football_truth_values())
    # print(find_girvan_newman_communities)
        

if __name__ == "__main__":
    main()
