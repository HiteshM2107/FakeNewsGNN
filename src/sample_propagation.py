import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from models_gnn import GAT

# Parameters
node_id = 123  # change as needed
model_name = "GAT"
checkpoint_path = f"checkpoints/{model_name}_seed42.pt"
k = 2  # number of hops

# Load graph
data = torch.load("data/processed/pyg_graph.pt")

# Load model
model = GAT(data.num_features, 16, 2)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Forward pass
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    predictions = logits.argmax(dim=1).cpu().numpy()

# Extract k-hop subgraph around node_id
subset, edge_index, mapping, _ = k_hop_subgraph(node_id, k, data.edge_index, relabel_nodes=True)
sub_data = data.subgraph(subset)
G = to_networkx(sub_data, to_undirected=True)

# Visualize
plt.figure(figsize=(8, 6))
colors = [predictions[n.item()] for n in subset]
nx.draw_networkx(
    G,
    node_color=colors,
    cmap=plt.cm.Set2,
    with_labels=True,
    node_size=500,
    font_size=10,
)
plt.title(f"{k}-Hop Neighborhood Propagation around Node {node_id} ({model_name})")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"results/sample_propagation_node{node_id}.png")
plt.close()

print(f"Saved propagation graph to results/sample_propagation_node{node_id}.png")
