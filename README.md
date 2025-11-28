# **CSE 841 – Artificial Intelligence – Final Project**
# Fake News Detection using Graph Neural Networks  

This project uses Graph Neural Networks (GCN, GraphSAGE, and GAT) to classify real vs. fake news articles from the PolitiFact subset of FakeNewsNet. By modeling semantic similarity between articles, the GNN learns contextual relationships that improve classification accuracy over standard text-based baselines.

---

# Project Pipeline

## 1. Data Preprocessing

Loads PolitiFact real/fake CSV files, cleans missing fields, merges title + text, and outputs:
`data/processed/posts.csv`


This file contains:

- text  
- title  
- label  
- content  

---

## 2. Graph Construction

Generates MiniLM embeddings, computes cosine similarity, and builds an undirected similarity graph.

Output: `data/processed/graph.pkl`

---

## 3. PyTorch Geometric Conversion

Converts the graph into a PyTorch Geometric `Data` object.

Output: `data/processed/pyg_graph.pt`

---

## 4. Baseline Model Training

Trains baseline ML models (Logistic Regression, SVM, Random Forest, Gradient Boosting).

Outputs:
`results/baselines.json`
`results/accuracy_vs_runtime.png`

---

## 5. GNN Training (GCN, GraphSAGE, GAT)

Each model is trained across several seeds. Outputs include checkpoints, reports, and learning curves.

Outputs:
`checkpoints/_seedXX.pt`
`results/_seedXX_report.json`
`results/_report.txt`
`results/_curves.png`
`results/summary.json`

---

## 6. Embedding Visualization (t-SNE)

Extracts hidden GNN embeddings and visualizes them using t-SNE.

Outputs:
`results/GCN_tsne.png`
`results/GraphSAGE_tsne.png`
`results/GAT_tsne.png`

---

# Project Structure

CSE841_FakeNewsGNN/
│
├── data/
│ ├── raw/
│ └── processed/
│ ├── posts.csv
│ ├── graph.pkl
│ └── pyg_graph.pt
│
├── src/
│ ├── build_graph.py
│ ├── to_pyg.py
│ ├── train_baselines.py
│ ├── train_gnn.py
│ ├── tsne_embeddings.py
│ └── models_gnn.py
│
├── results/
│
├── checkpoints/
│
├── notebooks/
│ └── FakeNews_GNN_Notebook.ipynb
│
├── environment.yml
└── README.md

---

# How to Run the Project

## Step 1 — Create the environment
`conda create -n cse841 python=3.10`
`conda activate cse841`

---

## Step 2 — Install dependencies
`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

`python -m pip install pandas numpy==1.26.4 scikit-learn networkx matplotlib sentence-transformers torch-geometric tqdm`

---

## Step 3 — Build the similarity graph

`python src/build_graph.py`


Output:
`data/processed/graph.pkl`

---

## Step 4 — Convert to PyTorch Geometric format

`python src/to_pyg.py`

Output:
`data/processed/pyg_graph.pt`

---

## Step 5 — Train GNN models
`python src/train_gnn.py --model gcn`
`python src/train_gnn.py --model sage`
`python src/train_gnn.py --model gat`

Results stored in:
`results/checkpoints/`

---

## Step 6 — Train baseline models
`python src/train_baselines.py`

---

## Step 7 — Visualize embeddings with t-SNE
`python src/tsne_embeddings.py --model GCN`
`python src/tsne_embeddings.py --model GraphSAGE`
`python src/tsne_embeddings.py --model GAT`

---

# Example Results

| Model      | Val Accuracy | Test Accuracy |
|------------|--------------|---------------|
| GCN        | 0.81         | 0.79          |
| GraphSAGE  | 0.85         | 0.83          |
| GAT        | 0.88         | 0.86          |
| SVM        | 0.78         | 0.76          |

---

# Key Insights

- GNNs outperform text-only baselines by leveraging inter-article relationships.
- GraphSAGE improves flexibility over GCN.
- GAT attends to important neighbors for best performance.
- t-SNE plots reveal clear separation between real and fake news clusters.

---

# Technologies Used

- Python 3.10  
- PyTorch  
- PyTorch Geometric  
- Sentence Transformers  
- scikit-learn  
- matplotlib  
- NetworkX  

---

# Acknowledgements

Dataset:

- FakeNewsNet (PolitiFact subset)

Key Papers:

- Kipf & Welling — GCN  
- Hamilton et al. — GraphSAGE  
- Veličković et al. — GAT  

---

# License

This project is provided for academic use as part of **CSE 841 Artificial Intelligence**.
