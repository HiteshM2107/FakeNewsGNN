import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, classification_report

from models_gnn import GCN, GraphSAGE, GAT


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


@torch.no_grad()
def predict(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    return out.argmax(dim=1)


def plot_curves(model_name, history, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    epochs = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{model_name} Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_curves.png"))
    plt.close()


def run_model(ModelClass, model_name, data, hidden_dim=64, lr=0.001, epochs=100):
    print(f"\n==============================")
    print(f"Training Model: {model_name}")
    print("==============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    num_features = data.x.size(1)
    num_classes = int(data.y.max().item() + 1)

    # Special-case GAT: has extra args
    if ModelClass.__name__ == "GAT":
        model = ModelClass(input_dim=num_features,
                           hidden_dim=hidden_dim,
                           output_dim=num_classes)
    else:
        model = ModelClass(input_dim=num_features,
                           hidden_dim=hidden_dim,
                           output_dim=num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0.0
    best_state = None

    history = {"loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, data.train_mask)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)

        history["loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | Train {train_acc:.4f} | Val {val_acc:.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_acc = evaluate(model, data, data.test_mask)
    print(f"\nBest Val Accuracy ({model_name}): {best_val:.4f}")
    print(f"Test Accuracy ({model_name}): {test_acc:.4f}\n")

    # Confusion matrix & classification report
    preds = predict(model, data).cpu()
    y_true = data.y.cpu()
    test_mask = data.test_mask.cpu()

    cm = confusion_matrix(y_true[test_mask], preds[test_mask])
    report = classification_report(y_true[test_mask], preds[test_mask], digits=4)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{model_name}_report.txt", "w") as f:
        f.write(f"{model_name} Confusion Matrix:\n{cm}\n\n")
        f.write(f"{model_name} Classification Report:\n{report}\n")

    print(f"{model_name} Confusion Matrix:\n{cm}")
    print(f"\n{model_name} Classification Report:\n{report}")

    # Plot curves
    plot_curves(model_name, history, out_dir="results")

    # Save best checkpoint for later (t-SNE, etc.)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{model_name.lower()}_best.pt")

    return best_val, test_acc


def main():
    print("Loading PyG graph...")
    data: Data = torch.load("data/processed/pyg_graph.pt")

    # Create split
    print("Generating train/val/test split...")
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

    # Run all three models
    gcn_val, gcn_test   = run_model(GCN,       "GCN",       data)
    sage_val, sage_test = run_model(GraphSAGE, "GraphSAGE", data)
    gat_val, gat_test   = run_model(GAT,       "GAT",       data)

    print("\n=========== FINAL COMPARISON ===========")
    print(f"GCN       → Val: {gcn_val:.4f}, Test: {gcn_test:.4f}")
    print(f"GraphSAGE → Val: {sage_val:.4f}, Test: {sage_test:.4f}")
    print(f"GAT       → Val: {gat_val:.4f}, Test: {gat_test:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
