import dgl
import numpy as np

# Load the Reddit dataset
dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

# Extract the number of nodes and edges
node_num = graph.number_of_nodes()
edge_num = graph.number_of_edges()

# Extract the edge list
src, dst = graph.edges()

# Combine source and destination nodes to form edges
edges = np.vstack((src.numpy(), dst.numpy())).T

# Create the formatted string
formatted_str = f"{node_num}\n{edge_num}\n"
for u, v in edges:
    formatted_str += f"{u} {v}\n"

# Save to a file
with open("cora_dataset.txt", "w") as f:
    f.write(formatted_str)

print("Cora dataset has been formatted and saved to 'cora_dataset.txt'")
