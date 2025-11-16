import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models_gnn import GCN, GraphSAGE


# Train & Evaluate Functions

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


def run_model(ModelClass, model_name, data, hidden_dim=64, lr=0.001, epochs=100):
    print(f"\n==============================")
    print(f"Training Model: {model_name}")
    print("==============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = ModelClass(
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        output_dim=int(data.y.max().item() + 1),
        dropout=0.5
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, data.train_mask)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} "
                f"| Train {train_acc:.4f} | Val {val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    test_acc = evaluate(model, data, data.test_mask)

    print(f"\nBest Val Accuracy ({model_name}): {best_val:.4f}")
    print(f"Test Accuracy ({model_name}): {test_acc:.4f}\n")

    return best_val, test_acc


# Main

def main():
    print("Loading graph...")
    data: Data = torch.load("data/processed/pyg_graph.pt")

    print("Running train/val/test split (70/15/15)...")
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

    # Train & compare models
    gcn_val, gcn_test = run_model(GCN, "GCN", data)
    sage_val, sage_test = run_model(GraphSAGE, "GraphSAGE", data)

    print("\n=========== FINAL COMPARISON ===========")
    print(f"GCN      → Val: {gcn_val:.4f}, Test: {gcn_test:.4f}")
    print(f"GraphSAGE → Val: {sage_val:.4f}, Test: {sage_test:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
