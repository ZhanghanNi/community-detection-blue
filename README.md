# Community Detection Blue Carleton College Winter 2025 Senior Project (COMPS) Repository 

### Yang Tan, Aidan Roessler, Tony Ni, Jake Jasmer

Also check out our project's website [here]() and our paper [here]().

## Installing Dependencies
Credit to [Python Documentation on Virtual Environments](https://docs.python.org/3/tutorial/venv.html) and Packages for the following instructions. Check it out if you want more information.

1. Create a virtual environment in the root directory of the project.
```bash
python -m venv community_detection_blue_comps_env
```

2. Activate the virtual environment.

On Mac run:
```bash
source community_detection_blue_comps_env/bin/activate
```
On Windows run:
```bash
community_detection_blue_comps_env\Scripts\activate
```

4. Install the required packages.
```bash
pip install -r requirements.txt
```

5. After you're done running our code, if you'd like to return to your normal environment, deactivate the virtual environment.
```bash
deactivate
```

## Running our Algorithms

Use `main.py` to run one of our implementations on one of our real world datasets. 

```bash
usage: python3 main.py [-h] 
        --algorithm {girvan_newman,louvain_method,bvns}
        --dataset {karate_club,college_football,celegans_neural,urban_movement_synthetic}
```

Note: runs all algorithms with no limit on the number of communities, instead all return communities that maximize modularity. Louvain runs for two iterations and BVNS runs with 100 iterations and a kmax of 3 for all datasets except for Karate Club and College Football where kmax is 4.

## File Structure
Each member of the team has their own directory with miscellaneous files. Please only run `main.py` as it has the capability to run all of our algorithms on all of our datasets. There is also a `Util` directory for shared utility functions used across implementations.

## Algorithms
- Girvan Newman: divisive algorithm that each iteration deletes the edge with the highest betweenness to create a set of communities (connected components) that maximize modularity. See `Aidan/girvan_newman_implementation/girvan_newman_implementation.py` for from scratch implementation.
- Louvain Method: uses a top down approach to merge nodes into communities that maximize modularity. See `Yang_Tan/louvain_method_implementation.py` for from scratch implementation.
- Basic Variable Neighborhood Search (BVNS): randomization approach to simulate gradient descent for forming communities with maximum modularity. See `Jake/bvns/bvns_implementation.py` for from scratch implementation.

## Datasets
- Karate Club: models the relationships of a karate club that split into two new clubs. Each individual is a node and edges between them represent relationships. Edges are weighted based on strength of relationship. Original dataset from [An Information Flow Model for Conflict and Fission in Small Groups by Wayne W. Zachary](https://www.jstor.org/stable/3629752).
- College Football: graph representation of the 2000 Division I season in college football which Girvan and Newman ran on their original implementation in their paper: [Community structure in social and
biological networks](https://www.pnas.org/doi/10.1073/pnas.122653799). In this graph nodes are college football teams and edges represent games between the teams the edges represent.
- Neural Network of C. Elegans: graph representation of the neural network of the C. Elegans worm. Each node is a neuron and edges represent synapses between neurons. Original labelings from Figure 3 of [Comprehensive analysis of the C. elegans connectome reveals novel circuits and functions of previously unstudied neurons](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002939) by Scott W. Emmons
<!-- TODO: Yang to fill in more detail-->
- Urban Movement Synthetic: synthetic dataset that models the movement of people in an urban area. 
<!-- TODO: Add PPI dataset -->
