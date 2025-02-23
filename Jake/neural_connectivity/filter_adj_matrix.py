import pandas as pd

# Load the adjacency matrix from a CSV file
adjacency_matrix = pd.read_csv('combined.csv', index_col=0)

# Get the index (row labels) of the DataFrame
row_labels = adjacency_matrix.index

# Filter columns to keep only those that have corresponding rows
filtered_adjacency_matrix = adjacency_matrix[row_labels]

# Save the filtered matrix to a new CSV file (optional)
filtered_adjacency_matrix.to_csv('symmetric_combined.csv')

# Display the filtered adjacency matrix
print(filtered_adjacency_matrix)
