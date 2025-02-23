import pandas as pd
import networkx as nx
from cdlib import algorithms as al
import sys
sys.path.append("../../Util")
import utils
sys.path.append("../bvns")
import bvns_implementation as bvns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "combined.csv"
df = pd.read_csv(file_path)

# Display basic information and the first few rows
df.info(), df.head()

# Extract neuron names
neurons = df.columns[1:]  # Exclude the first column which contains row labels

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for index, row in df.iterrows():
    pre_synaptic = row.iloc[0]  # The first column contains the pre-synaptic neuron name
    for post_synaptic, weight in row.iloc[1:].items():  # Remaining columns are post-synaptic neurons
        if pd.notna(weight):  # Only add edges for non-NaN values
            G.add_edge(pre_synaptic, post_synaptic, weight=weight)

# Basic graph stats
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print("Edges: ", num_edges)
print("Nodes: ", num_nodes)

# Define communities as a dictionary where keys are community names and values are sets of nodes
communities_list = [
    set([
        "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DB01", "DB02", "DB03", "DB04", "DB05", "DB06",
        "AS01", "AS02", "AS03", "AS04", "AS05", "AS06", "AS07", "AS08", "AS09", "DD01", "DD02", "DD03", "DD04", "DD05",
        "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VB02", "VB03", "VB04", "VB05",
        "VB06", "VB07", "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VC01", "VC02", "VC03",
        "dBWML7", "dBWML8", "dBWML9", "dBWML10", "dBWML11", "dBWML12", "dBWML13", "dBWML14", "dBWML15", "dBWML16", "dBWML17",
        "dBWML18", "dBWML19", "dBWMR8", "dBWMR9", "dBWMR10", "dBWMR11", "dBWMR12", "dBWMR13", "dBWMR14", "dBWMR15",
        "dBWMR16", "dBWMR17", "dBWMR18", "dBWMR19", "dBWMR20", "vBWML8", "vBWML9", "vBWML10", "vBWML11", "vBWML12",
        "vBWML13", "vBWML14", "vBWML15", "vBWML16", "vBWML17", "vBWML18", "vBWMR10", "vBWMR11", "vBWMR12", "vBWMR13",
        "vBWMR14", "vBWMR15", "vBWMR16", "vBWMR17", "vBWMR18", "vBWMR19"
    ]),
    set([
        "IL1DL", "IL1DR", "IL1L", "IL2R", "IL1VL", "IL1VR", "IL2DL", "IL2DR", "IL2VL", "IL2VR", "SDQL", "SDQR", "RIPL", "RIPR",
        "RMED", "RMEL", "RMER", "RMEV", "URADL", "URADR", "URAVL", "URAVR", "SABD", "SABVL", "SABVR", "SIADL", "SIADR",
        "SMBDL", "SMBDR", "dBWML1", "dBWML2", "dBWML3", "dBWML4", "dBWML5", "dBWML6", "dBWMR1", "dBWMR2", "dBWMR3", "dBWMR4",
        "dBWMR5", "dBWMR6", "dBWMR7", "vBWML1", "vBWML2", "vBWML3", "vBWML4", "vBWMR1", "vBWMR2", "vBWMR3", "vBWMR4"
    ]),
    set([
        "ADFL", "ADFR", "ADLL", "ADLR", "ASEL", "ASER", "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "AWAL", "AWAR",
        "AWBL", "AWBR", "AWCL", "AWCR", "AFDL", "AFDR", "AIAL", "AIAR", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR",
        "ADAL", "ADAR", "AIML", "AINL", "AINR", "RIR", "RIML", "RIMR", "SAADL", "SAADR", "SAAVL", "SAAVR", "SMBVL",
        "SMBVR", "VB01"
    ]),
    set([
        "IL2L", "IL2R", "OLQDL", "OLQDR", "OLQVL", "OLQVR", "OLLL", "OLLR", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "ALNL",
        "ALNR", "PLNL", "PLNR", "BAGL", "BAGR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR", "AUAL", "AUAR",
        "ADEL", "ADER", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RIGL", "RIGR", "RIH", "RIS", "RMGL", "RMGR",
        "URBL", "URBR", "AVEL", "AVER", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMHL", "RMHR", "RIVL",
        "RIVR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "vBWML5", "vBWML6", 
        "vBWML7", "vBWMR5", "vBWMR6", "vBWMR7", "vBWMR8", "vBWMR9"
    ]),
    set(["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR",
         "pm1", "pm2D", "pm2VL", "pm2VR", "pm3D", "pm3VL", "pm3VR", "pm4D", "pm4VL", "pm4VR", "pm5D", "pm5VL", "pm5VR", "pm6D", 
         "pm6VL", "pm6VR", "pm7D", "pm7VL", "pm7VR", "pm8", "mc1DL", "mc1DR", "mc1V", "mc2DL", "mc2DR", "mc2V", "mc3DL", "mc3DR", "mc3V"
         ]),
    set(["vm1AL", "vm1AR", "vm1PL", "vm1PR"]),
    set(["mu_intL", "mu_intR", "mu_sph", "mu_anal"]),
    set(["um2AL", "um2AR", "um1AL", "um1AR", "um1PL", "um1PR", "um2PL", "um2PR", "um2AL"]),
    set(["ALML", "ALMR", "AVM", "PVM", "FLPL", "FLPR", "BDUL", "BDUR", "PVDL", "PVDR", "AQR", "PQR", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR", "PHCL", "PHCR",
         "ALA", "RIFL", "RIFR", "RMFL", "RMFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL", "DVA", "DVB", "DVC", "PVNL", "PVNR", "PVPL", "PVPR",
         "PVR", "PVT", "PVWL", "PVWR", "RID", "PVCL", "PVCR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "LUAL", "LUAR", "DA08", "DA09", "PDA", "DB07", "AS10",
         "AS11", "PDB", "DD06", "VA11", "VA12", "VB08", "VB09", "VB10", "VB11", "VD10", "VD11", "VD12", "VD13", "VC06", "PLML", "PLMR", "dBWML20", "dBWML21", "dBWML22",
         "dBWML23", "dBWML24", "dBWMR21", "dBWMR22", "dBWMR23", "dBWMR24", "vBWML19", "vBWML20", "vBWML21", "vBWML22", "vBWML23", "vBWMR20", "vBWMR21", "vBWMR22",
         "vBWMR23", "vBWMR24", "hyp", "int"]),
    set(["ASIL", "ASJR", "ASKL", "ASKR", "AIMR", "AVFL", "AVFR", "PVQL", "PVQR", "HSNL", "HSNR", "VC04", "VC05", "vm2AL", "vm2AR", "vm2PL", "vm2PR"]),
    # Trying an extra community of unused nodes
    set(['GLRVR', 'hmc', 'g2L', 'g1AL', 'IL1R', 'GLRR', 'bm', 'e3D', 'e2VL', 'g2R', 'CEPshVR', 'exc_gl', 'CEPshDL', 'e2D', 'GLRVL', 'g1AR', 'e3VR', 'GLRDR', 'GLRDL', 'e2VR', 'CEPshVL', 'exc_cell', 'CANR', 'ASJL', 'GLRL', 'e3VL', 'CEPshDR', 'CANL', 'g1p'])
]

all_nodes = set(G.nodes)
community_nodes = set().union(*communities_list)
print("Nodes not in graph:")
print(community_nodes - all_nodes)  # Should be an empty set

print("Nodes not in communities:")
all_nodes_in_partition = set.union(*communities_list)
if all_nodes_in_partition != set(G.nodes):
    print("Mismatch in covered nodes:", set(G.nodes) - all_nodes_in_partition)

# Compute modularity
modularity = nx.algorithms.community.modularity(G, communities_list)

print(f"Modularity Score: {modularity}")

