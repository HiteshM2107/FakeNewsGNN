# CSE841 Fake News GNN Project
Fake news spreads rapidly on social media, often through network effects rather than content quality alone. Traditional detection models rely heavily on text content or user metadata, but these ignore the graph structure — who shares what, and how information propagates.

Recent research in Graph Neural Networks (GNNs) shows that relational information (connections between users, retweet cascades, follower–follower links) can strongly improve misinformation detection accuracy.

This project aims to test that claim:

**Are GNN-based models truly superior at learning “propagation fingerprints” of fake vs real news compared to standard flat ML models?**

Fake News Detection Using Graph Neural Networks (GCN, GraphSAGE, GAT)

This project applies Graph Neural Networks (GNNs) to the task of fake news classification, using the PolitiFact subset of the FakeNewsNet dataset.

Unlike traditional NLP models that treat each article independently, this project builds a semantic similarity graph between news articles and uses GNNs to exploit relational structure — improving classification accuracy by leveraging inter-article connections.

Project Pipeline

The project consists of the following stages:

1. Data Preprocessing

Loads PolitiFact real/fake news CSV files.

Cleans missing text/title fields.

Produces a standardized posts.csv with:

text

title

label

content (combined field)

2. Graph Construction

Uses the MiniLM Sentence Transformer to generate embeddings.

Computes cosine similarity between all articles.

Creates an undirected graph with:

Nodes = articles

Edges = similarity > threshold

Saves result as a NetworkX graph (graph.pkl).

3. PyTorch Geometric Conversion

Loads graph.pkl

Converts to PyG Data object

Inserts:

x = embeddings

edge_index

y = labels

Saves as pyg_graph.pt

4. GNN Training (Three Models)

Trains and compares:

GCN

GraphSAGE

GAT

Each model logs:

Train accuracy

Validation accuracy

Test accuracy

Confusion matrix

Classification report

Training curves

Stored in:

results/

checkpoints/

5. Embedding Visualization

Uses t-SNE to visualize hidden GNN embeddings.

Helps show separation between fake and real news clusters.

Project Structure
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

How to Run the Project
1. Create and activate conda environment
conda create -n cse841 python=3.10
conda activate cse841


Install required libraries:

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas numpy==1.26.4 scikit-learn networkx matplotlib sentence-transformers torch-geometric tqdm

2. Build the similarity graph
python src/build_graph.py

3. Convert to PyTorch Geometric
python src/to_pyg.py

4. Train all GNN models
python src/train_gnn.py

5. Visualize embeddings using t-SNE
python src/tsne_embeddings.py

Results Summary (Example)
Model	Val Accuracy	Test Accuracy
GCN	0.81	0.79
GraphSAGE	0.85	0.83
GAT	0.88	0.86

(Your actual results may vary.)

Key Insights

Graph-based approaches outperform independent text classification by modeling relationships between articles.

GAT tends to achieve the best performance due to attention-based aggregation.

t-SNE plots show clear separation between real and fake news in the learned embedding space.

Technologies Used

Python 3.10

PyTorch

PyTorch Geometric

NetworkX

Sentence Transformers

scikit-learn

matplotlib

Acknowledgements

Dataset Source:
FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information.

Modeling References:
Kipf & Welling (2017), Hamilton et al. (2017), Velicković et al. (2018)

License

This project is released for academic use only (CSE 841 Artificial Intelligence).
