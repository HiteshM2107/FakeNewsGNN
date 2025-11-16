import itertools
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models_gnn import GCN  # We can switch to GraphSAGE or GAT if we want


def train(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    correct = (preds[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    return correct / total if total > 0 else 0.0


def main():
    print("Loading data...")
    data: Data = torch.load("data/processed/pyg_graph.pt")

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    num_features = data.x.size(1)
    num_classes = int(data.y.max().item() + 1)

    hidden_dims = [32, 64, 128]
    dropouts = [0.3, 0.5]
    lrs = [0.001, 0.005]

    best_cfg = None
    best_val = 0.0

    print("Starting hyperparameter search for GCN...")
    for h, d, lr in itertools.product(hidden_dims, dropouts, lrs):
        print(f"\nConfig: hidden={h}, dropout={d}, lr={lr}")

        model = GCN(num_features, h, num_classes, dropout=d).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # Short training (e.g., 50 epochs)
        for epoch in range(1, 51):
            loss = train(model, data, optimizer, data.train_mask)
            if epoch % 10 == 0 or epoch == 1:
                val_acc = evaluate(model, data, data.val_mask)
                print(f"  Epoch {epoch:03d} | Loss {loss:.4f} | Val {val_acc:.4f}")

        val_acc = evaluate(model, data, data.val_mask)
        print(f"Final Val Acc for this config: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_cfg = (h, d, lr)

    print("\nBest Config:")
    print(f"hidden_dim={best_cfg[0]}, dropout={best_cfg[1]}, lr={best_cfg[2]}")
    print(f"Best Val Accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()
