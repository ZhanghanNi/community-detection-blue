This folder contains work for making nx graphs with c elegans neuronal data. The one that I'd like
to use is SI7_herm.csv which is made into an nx graph in csv_to_nx_graph.py and has labels. It
comes from the paper found here: https://wormwiring.org/papers/Cook%20et%20al%202019.pdf

I think the closest I'll get to "truth values" is the information found in the SI 6 file which is
lists of different types of cells (SN stands for sensory neuron and IN stands for interneuron). 
Oddly enough, we should not expect communities of the same type of neurons because there's a
directional flow of information. We should expect a mix of SNs and INs in each community if 
community detection can be used to indicate neuronal circuits.
