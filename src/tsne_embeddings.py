import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from models_gnn import GCN, GraphSAGE, GAT


def get_model(model_name, input_dim, hidden_dim, output_dim):
    if model_name == "GCN":
        return GCN(input_dim, hidden_dim, output_dim)
    elif model_name == "GraphSAGE":
        return GraphSAGE(input_dim, hidden_dim, output_dim)
    elif model_name == "GAT":
        return GAT(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main(model_name="GAT", hidden_dim=64):
    print(f"t-SNE for model: {model_name}")

    data: Data = torch.load("data/processed/pyg_graph.pt")
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item() + 1)

    ckpt_path = f"checkpoints/{model_name.lower()}_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. "
                                f"Run train_gnn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = get_model(model_name, num_features, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Get hidden embeddings
    with torch.no_grad():
        emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
        labels = data.y.cpu().numpy()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        mask = labels == c
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=10,
            alpha=0.7,
            label=f"Class {c}",
        )

    plt.title(f"{model_name} Node Embeddings (t-SNE)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{model_name}_tsne.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    main(model_name="GCN")
