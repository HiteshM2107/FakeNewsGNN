import torch
from torch.optim import Adam
from sklearn.metrics import f1_score

def train_gnn_model(model, data, model_name="model"):
    optimizer = Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.NLLLoss()

    x, edge_index, y = data.x, data.edge_index, data.y

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        f1 = f1_score(y.cpu(), pred.cpu(), average="macro")

        if epoch % 10 == 0:
            print(f"[{model_name}] Epoch {epoch}, Loss: {loss:.4f}, F1: {f1:.4f}")