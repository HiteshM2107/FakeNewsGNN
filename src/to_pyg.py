import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def load_graph():
    with open("data/processed/graph.pkl", "rb") as f:
        G = pickle.load(f)
    return G

def to_pyg():
    print("Loading NetworkX graph...")
    G = load_graph()

    # Convert to basic PyG structure (node attrs, edge list)
    print("Converting to PyTorch Geometric Data...")
    data = from_networkx(G)

    # We MUST manually create feature matrix "x" because similarity graph nodes have no features yet
    print("Building feature matrix from sentence embeddings...")

    # Load same embeddings used earlier for consistency
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv("data/processed/posts.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    contents = df["content"].tolist()
    embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=True)

    # Convert to torch tensor
    x = torch.tensor(embeddings, dtype=torch.float)

    # Labels from NetworkX node attributes
    y = torch.tensor([G.nodes[n]["label"] for n in G.nodes()], dtype=torch.long)

    # Assign tensors into data object
    data.x = x
    data.y = y

    print("Data object summary:")
    print(data)

    # Save
    torch.save(data, "data/processed/pyg_graph.pt")
    print("\nSaved PyTorch Geometric graph to data/processed/pyg_graph.pt")

if __name__ == "__main__":
    to_pyg()