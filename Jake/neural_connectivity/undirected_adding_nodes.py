import pandas as pd

# Load the adjacency matrix (assuming a CSV with row labels)
adj_matrix = pd.read_csv('combined.csv', index_col=0)

# Find missing rows (since we have 7 extra columns)
missing_rows = set(adj_matrix.columns) - set(adj_matrix.index)

# Add missing rows with zeros
for row in missing_rows:
    adj_matrix.loc[row] = 0  # Add row with all zeros

# Reorder rows to match columns (ensuring a square matrix)
adj_matrix = adj_matrix.reindex(index=adj_matrix.columns, fill_value=0)

# Make the matrix symmetric (undirected graph)
adj_matrix = adj_matrix.where(adj_matrix > 0, adj_matrix.T)  # Copy reverse edges

# Save the symmetric adjacency matrix
adj_matrix.to_csv('undirected_added_nodes.csv')

# Print the final shape to confirm it's square
print(adj_matrix.shape)

adj_matrix = adj_matrix.fillna(0)  # Replace NaNs with 0
num_edges = (adj_matrix.to_numpy() != 0).sum() // 2  # Count nonzero edges, divide by 2 for undirected graph
print(f"Number of edges: {num_edges}")
