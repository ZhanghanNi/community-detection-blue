"""
** Run this file to run our Girvan-Newman implementation **

Wrapper file around our Girvan-Newman algorithm implementation
"""
import networkx as nx
import girvan_newman as gn
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

def main():
    """ Main function that runs our implementation on test data for now"""

    # TODO: rethink name and if we should pass data in here
    # For now we are just passing in the karate club graph
    print("calling main!")

    G = nx.karate_club_graph()
    

    # TODO: call girvan newman here
    gn.girvan_newman(G)


if __name__ == "__main__":
    main()
