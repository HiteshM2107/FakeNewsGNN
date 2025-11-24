import torch
import torch.nn.functional as F
import time
import random
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import json
from models_gnn import GCN, GraphSAGE, GAT

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Load the PyG graph
data = torch.load("../data/processed/pyg_graph.pt")

# Create results directories
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Train-validation-test split (same across seeds)
def get_split(num_nodes):
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    train_idx = torch.tensor(indices[:train_size])
    val_idx = torch.tensor(indices[train_size:train_size + val_size])
    test_idx = torch.tensor(indices[train_size + val_size:])
    return train_idx, val_idx, test_idx

# Training function
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def test(model, data, split_idx):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    acc = accuracy_score(data.y[split_idx].cpu(), pred[split_idx].cpu())
    return acc, pred

# Main loop for multiple models and seeds
models = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT
}

results_summary = {}
times = {}

for model_name, model_class in models.items():
    print(f"\n--- Training {model_name} ---")
    val_scores = []
    test_scores = []
    run_times = []

    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        train_idx, val_idx, test_idx = get_split(data.num_nodes)
        model = model_class(data.num_features, 16, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        start_time = time.time()
        best_val = 0
        for epoch in range(1, 201):
            loss = train(model, data, train_idx, optimizer)
            val_acc, _ = test(model, data, val_idx)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), f"checkpoints/{model_name}_seed{seed}.pt")
        end_time = time.time()

        model.load_state_dict(torch.load(f"checkpoints/{model_name}_seed{seed}.pt"))
        val_acc, _ = test(model, data, val_idx)
        test_acc, pred = test(model, data, test_idx)

        val_scores.append(val_acc)
        test_scores.append(test_acc)
        run_times.append(end_time - start_time)

        report = classification_report(data.y[test_idx].cpu(), pred[test_idx].cpu(), output_dict=True)
        with open(f"results/{model_name}_seed{seed}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"Seed {seed} — Val: {val_acc:.4f}, Test: {test_acc:.4f}, Time: {end_time - start_time:.2f}s")

    # Aggregate results
    results_summary[model_name] = {
        "val_acc_mean": np.mean(val_scores),
        "val_acc_std": np.std(val_scores),
        "test_acc_mean": np.mean(test_scores),
        "test_acc_std": np.std(test_scores)
    }
    times[model_name] = run_times

# Save results
with open("results/summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Plot accuracy vs runtime
plt.figure()
for model_name in models:
    plt.plot(times[model_name], label=f"{model_name} runtime (s)")
plt.xlabel("Run Index (Seed)")
plt.ylabel("Runtime (s)")
plt.title("Runtime per Model across Seeds")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_vs_runtime.png")
plt.close()

print("\n Training complete. Results and plots saved to results/")
