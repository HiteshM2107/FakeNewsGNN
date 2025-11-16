import pandas as pd
import os

def preprocess_politifact(base_path="data/raw/politifact/"):
    fake_path = os.path.join(base_path, "fake", "politifact_fake.csv")
    real_path = os.path.join(base_path, "real", "politifact_real.csv")

    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    # Normalize columns (different mirrors use different naming)
    df_fake = df_fake.rename(columns={col: col.lower() for col in df_fake.columns})
    df_real = df_real.rename(columns={col: col.lower() for col in df_real.columns})

    # Add labels
    df_fake["label"] = 1
    df_real["label"] = 0

    # Select required columns
    cols = ["id", "title", "text", "label"]
    df_fake = df_fake[[c for c in cols if c in df_fake.columns]]
    df_real = df_real[[c for c in cols if c in df_real.columns]]

    df = pd.concat([df_fake, df_real], ignore_index=True)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/posts.csv", index=False)

    print("Saved data/processed/posts.csv")
    print("Total posts:", len(df))

if __name__ == "__main__":
    preprocess_politifact()