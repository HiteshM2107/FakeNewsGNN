import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle

def build_graph():
    print("Loading posts...")
    df = pd.read_csv("data/processed/posts.csv")

    print("Preparing content column...")

    # Handle missing title/text columns safely
    if "title" in df.columns:
        title_col = df["title"].astype(str).fillna("")
    else:
        title_col = pd.Series([""] * len(df))

    if "text" in df.columns:
        text_col = df["text"].astype(str).fillna("")
    else:
        text_col = pd.Series([""] * len(df))

    df["content"] = (title_col + " " + text_col).str.strip()

    # Save updated posts.csv with content column
    df.to_csv("data/processed/posts.csv", index=False)
    print("Updated posts.csv saved with new 'content' column.")

    print("Loading embedding model (MiniLM)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    contents = df["content"].tolist()
    embeddings = model.encode(contents, convert_to_numpy=True, show_progress_bar=True)

    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)

    print("Building similarity graph...")
    G = nx.Graph()

    # Add nodes with labels
    for idx, row in df.iterrows():
        G.add_node(idx, label=int(row["label"]))

    # Add edges based on similarity
    threshold = 0.6
    num_nodes = len(df)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i, j]))

    # Save using pickle
    print("Saving graph to data/processed/graph.pkl")
    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/graph.pkl", "wb") as f:
        pickle.dump(G, f)

    print("\nGraph saved successfully!")
    print("Total nodes:", G.number_of_nodes())
    print("Total edges:", G.number_of_edges())

if __name__ == "__main__":
    build_graph()
