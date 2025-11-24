import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from models_gnn import GAT

# Load data and model
node_id = 123
data = torch.load("data/processed/pyg_graph.pt")
model = GAT(data.num_features, 16, 2)
model.load_state_dict(torch.load("checkpoints/GAT_seed42.pt"))
model.eval()

# Run one layer manually to get attention weights
with torch.no_grad():
    _, (edge_idx, alpha) = model.conv1(data.x, data.edge_index, return_attention_weights=True)

# Extract neighbors and attention
edge_index = edge_idx.cpu().numpy()
alpha = alpha.cpu().numpy()
G = to_networkx(data, to_undirected=True)
neighbors = list(G.neighbors(node_id))

attn_dict = {}
for i, (src, tgt) in enumerate(edge_index.T):
    if src == node_id and tgt in neighbors:
        attn_dict[tgt] = alpha[i].mean()

# Plot heatmap
labels = list(attn_dict.keys())
weights = [attn_dict[n] for n in labels]
plt.figure(figsize=(10, 0.5))
plt.imshow([weights], cmap='viridis', aspect='auto')
plt.yticks([])
plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90)
plt.colorbar(label="Attention Weight")
plt.title(f"GAT Attention from Node {node_id} to Neighbors")
plt.tight_layout()
plt.savefig(f"results/gat_attention_node{node_id}.png")
plt.close()

print(f"Saved attention heatmap to results/gat_attention_node{node_id}.png")