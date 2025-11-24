import os
import glob
import argparse

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.data import Data

from models_gnn import GCN, GraphSAGE, GAT


def get_model(model_name: str, input_dim: int, hidden_dim: int, output_dim: int):
    """Return the correct GNN model class based on name."""
    if model_name == "GCN":
        return GCN(input_dim, hidden_dim, output_dim)
    elif model_name == "GraphSAGE":
        return GraphSAGE(input_dim, hidden_dim, output_dim)
    elif model_name == "GAT":
        return GAT(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def find_checkpoint(model_name: str) -> str:
    """
    Find a checkpoint file for the given model.

    It looks for files matching:
        checkpoints/{model_name}_seed*.pt
    and returns the first match (sorted).
    """
    pattern = os.path.join("checkpoints", f"{model_name}_seed*.pt")
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoints found matching pattern: {pattern}")

    ckpt_path = sorted(matches)[0]
    print(f"Using checkpoint: {ckpt_path}")
    return ckpt_path


def detect_hidden_dim_from_state(state_dict: dict, model_name: str) -> int:
    """
    Infer the hidden dimension from the checkpoint state dict.

    For GAT:
        conv1.att_src shape = [1, heads, hidden_dim]
    For GCN/GraphSAGE:
        conv1.*.weight shape = [hidden_dim, input_dim]
    """
    if model_name == "GAT":
        # Prefer using attention tensor to get hidden_dim reliably
        for k, v in state_dict.items():
            if "conv1.att_src" in k:
                # v.shape = [1, heads, hidden_dim]
                if v.dim() == 3:
                    hidden_dim = v.shape[2]
                    print(f"Detected GAT hidden_dim = {hidden_dim} from '{k}'")
                    return hidden_dim

        # Fallback: use conv1.lin.weight (rows = heads * hidden_dim)
        for k, v in state_dict.items():
            if "conv1" in k and "weight" in k and v.dim() == 2:
                total_out = v.shape[0]
                heads = 4  # matches your GATConv(..., heads=4, ...)
                hidden_dim = total_out // heads
                print(
                    f"Fallback for GAT: total_out = {total_out}, "
                    f"assumed heads = {heads} → hidden_dim = {hidden_dim} from '{k}'"
                )
                return hidden_dim

        raise RuntimeError("Could not detect hidden_dim for GAT from checkpoint.")
    else:
        # GCN / GraphSAGE: hidden_dim is just the rows of conv1 weight matrix
        for k, v in state_dict.items():
            if "conv1" in k and "weight" in k and v.dim() == 2:
                hidden_dim = v.shape[0]
                print(f"Detected hidden_dim = {hidden_dim} from '{k}'")
                return hidden_dim

        raise RuntimeError("Could not detect hidden_dim from checkpoint for non-GAT model.")


def run_tsne(model_name: str, data_path: str = "../data/processed/pyg_graph.pt"):
    print(f"=== Running t-SNE for model: {model_name} ===")

    # Load PyG graph
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PyG graph not found at: {data_path}")
    data: Data = torch.load(data_path)
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item() + 1)
    print(f"Loaded graph with {data.num_nodes} nodes, {data.num_edges} edges.")
    print(f"Node feature dim = {num_features}, num_classes = {num_classes}")

    # Find and load checkpoint
    ckpt_path = find_checkpoint(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    state = torch.load(ckpt_path, map_location=device)

    # Auto-detect hidden_dim from checkpoint
    hidden_dim = detect_hidden_dim_from_state(state, model_name)

    # Build model with correct hidden_dim
    model = get_model(model_name, num_features, hidden_dim, num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    # Get node embeddings from model
    with torch.no_grad():
        # Assumes your GNN classes implement `get_embeddings(x, edge_index)`
        emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
        labels = data.y.cpu().numpy()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    emb_2d = tsne.fit_transform(emb)

    # Plot
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
    out_path = os.path.join("results", f"{model_name}_tsne.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved t-SNE plot to: {out_path}")
    print("=== Done ===")


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization for GNN node embeddings.")
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=["GCN", "GraphSAGE", "GAT"],
        help="Which GNN model to visualize.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/processed/pyg_graph.pt",
        help="Path to the saved PyG Data object.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tsne(model_name=args.model, data_path=args.data_path)
