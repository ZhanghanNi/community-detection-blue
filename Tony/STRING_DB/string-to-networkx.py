import networkx as nx

def create_ppi_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        next(f)  
        for line in f:
            line = line.strip()
            if not line:
                continue         
            protein1, protein2, combined_score = line.split()    
            combined_score = float(combined_score)          
            G.add_edge(protein1, protein2, weight=combined_score)
    return G

if __name__ == "__main__":
    ppi_file = "/Users/zhanghanni/Desktop/comps/STRING/195.protein.links.v12.0.txt"
    ppi_graph = create_ppi_graph(ppi_file)
    for i, edge_data in enumerate(ppi_graph.edges(data=True)):
        if i == 5: 
            break
        print(edge_data)