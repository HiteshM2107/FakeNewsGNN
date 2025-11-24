import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import json
import os

# Load post data
df = pd.read_csv("data/processed/posts.csv")
X = df["content"]
y = df["label"]

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)

# Models to evaluate
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4)
    }
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# Save results
os.makedirs("results", exist_ok=True)
with open("results/baselines.json", "w") as f:
    json.dump(results, f, indent=2)
