from src.preprocess import load_fnn_data, clean_text
from src.build_graph import build_graph_ptg
from src.baselines import run_baselines
from src.train_utils import train_gnn_model
from src.models_gnn import GCN, GraphSAGE, GAT

def main():

    print("Loading FakeNewsNet data...")
    df_posts, df_users, edges = load_fnn_data()

    print("Building graph...")
    data = build_graph_ptg(df_posts, df_users, edges)

    print("Running baselines...")
    run_baselines(df_posts, df_users)

    print("Training GCN model...")
    gcn = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)
    train_gnn_model(gcn, data, model_name="GCN")

    print("Training GraphSAGE model...")
    sage = GraphSAGE(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)
    train_gnn_model(sage, data, model_name="GraphSAGE")

    print("Training GAT model...")
    gat = GAT(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)
    train_gnn_model(gat, data, model_name="GAT")

if __name__ == "__main__":
    main()
