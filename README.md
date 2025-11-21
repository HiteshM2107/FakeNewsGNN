# CSE841 Fake News GNN Project
Fake news spreads rapidly on social media, often through network effects rather than content quality alone. Traditional detection models rely heavily on text content or user metadata, but these ignore the graph structure — who shares what, and how information propagates.

Recent research in Graph Neural Networks (GNNs) shows that relational information (connections between users, retweet cascades, follower–follower links) can strongly improve misinformation detection accuracy.

This project aims to test that claim:

**Are GNN-based models truly superior at learning “propagation fingerprints” of fake vs real news compared to standard flat ML models?**

Fake News Detection Using Graph Neural Networks (GCN, GraphSAGE, GAT)

This project applies Graph Neural Networks (GNNs) to the task of fake news classification, using the PolitiFact subset of the FakeNewsNet dataset.

Unlike traditional NLP models that treat each article independently, this project builds a semantic similarity graph between news articles and uses GNNs to exploit relational structure — improving classification accuracy by leveraging inter-article connections.

## Project Pipeline

The project consists of the following stages.

### 1. Data Preprocessing
- Loads PolitiFact real/fake CSV files.
- Cleans missing text/title fields.
- Produces a standardized `posts.csv` with:
  - `text`
  - `title`
  - `label`
  - `content` (combined field).

### 2. Graph Construction
- Uses the MiniLM Sentence Transformer to generate embeddings.
- Computes cosine similarity between all articles.
- Builds an undirected graph where:
  - Nodes = articles  
  - Edges = similarity > threshold  
- Saves the graph as a NetworkX object: `graph.pkl`.

### 3. PyTorch Geometric Conversion
- Loads `graph.pkl`.
- Converts it to a PyTorch Geometric `Data` object.
- Adds:
  - `x` = embeddings  
  - `edge_index`  
  - `y` = labels  
- Saves the PyG graph as `pyg_graph.pt`.

### 4. GNN Training (GCN, GraphSAGE, GAT)
Trains and compares three GNN architectures:
- GCN  
- GraphSAGE  
- GAT  

For each model, the training script:
- Splits nodes into train (70%), validation (15%), and test (15%).
- Tracks:
  - Training loss
  - Training accuracy
  - Validation accuracy
  - Test accuracy
- Saves:
  - Confusion matrix
  - Classification report
  - Training curves (loss / accuracy)
  - Best model checkpoint

Outputs are written to:
- `results/`
- `checkpoints/`

### 5. Embedding Visualization
- Uses t-SNE to reduce hidden GNN embeddings to 2D.
- Produces a scatter plot showing separation between fake vs. real news.

---

## Project Structure

```text
CSE841_FakeNewsGNN/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── posts.csv
│       ├── graph.pkl
│       └── pyg_graph.pt
│
├── src/
│   ├── build_graph.py
│   ├── to_pyg.py
│   ├── train_gnn.py
│   ├── tsne_embeddings.py
│   └── models_gnn.py
│
├── results/
├── checkpoints/
│
├── FakeNews_GNN_Notebook.ipynb
├── environment.yml
└── README.md

## How to Run the Project

### Step 1 — Create and activate a conda environment
```bash
conda create -n cse841 python=3.10
conda activate cse841
```

### Step 2 — Install dependencies
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas numpy==1.26.4 scikit-learn networkx matplotlib sentence-transformers torch-geometric tqdm
```

### Step 3 — Build the similarity graph
```bash
python src/build_graph.py
```
This will:
- Load `data/processed/posts.csv`
- Generate sentence embeddings
- Compute cosine similarity
- Build the similarity graph
- Save `data/processed/graph.pkl`

### Step 4 — Convert to PyTorch Geometric format
```bash
python src/to_pyg.py
```
This will:
- Load `graph.pkl`
- Load `posts.csv`
- Build `data.x`, `data.edge_index`, `data.y`
- Save `data/processed/pyg_graph.pt`

### Step 5 — Train all GNN models (GCN, GraphSAGE, GAT)
```bash
python src/train_gnn.py
```
This script will:
- Create train/validation/test node splits
- Train **GCN**, **GraphSAGE**, and **GAT**
- Save:
  - Training curves in `results/*.png`
  - Confusion matrices and classification reports in `results/*_report.txt`
  - Best model checkpoints in `checkpoints/*.pt`
- Print a final comparison table with validation and test accuracy for each model

### Step 6 — Visualize t-SNE embeddings
```bash
python src/tsne_embeddings.py
```
This will:
- Load the best checkpoint (default: **GAT**)
- Extract hidden embeddings using `get_embeddings`
- Run t-SNE (2D)
- Save the plot to `results/GAT_tsne.png` (or the corresponding model name)

---

## Results Summary (Example)

| Model      | Val Accuracy | Test Accuracy |
|------------|--------------|---------------|
| GCN        | 0.81         | 0.79          |
| GraphSAGE  | 0.85         | 0.83          |
| GAT        | 0.88         | 0.86          |

*Actual results may vary with threshold, hyperparameters, and data.*

---

## Key Insights

- Graph-based approaches outperform independent text classification by modeling inter-article relationships.
- GraphSAGE improves on GCN via flexible neighborhood aggregation.
- GAT often performs best by attending to important neighbors.
- t-SNE visualizations reveal clear separation between classes in learned embeddings.

---

## Technologies Used

- Python 3.10  
- PyTorch  
- PyTorch Geometric  
- NetworkX  
- Sentence Transformers  
- scikit-learn  
- matplotlib

---

## Acknowledgements

**Dataset**  
FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information (PolitiFact subset).

**Key Modeling References**  
Kipf & Welling — Semi-Supervised Classification with Graph Convolutional Networks (GCN)  
Hamilton et al. — Inductive Representation Learning on Large Graphs (GraphSAGE)  
Velicković et al. — Graph Attention Networks (GAT)

---

## License

This project is provided for academic use only as part of **CSE 841 Artificial Intelligence**.
