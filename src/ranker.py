import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_embeddings(split="train"):
    """Loads article embeddings dictionary."""
    path = os.path.join(PROCESSED_DIR, f"embeddings_{split}.npy")
    return np.load(path, allow_pickle=True).item()

def compute_user_vector(history_str, embeddings_dict, default_dim=384):
    """
    Calculates a user vector by averaging embeddings of clicked news in history.
    """
    if pd.isna(history_str) or not str(history_str).strip():
        return np.zeros(default_dim)

    history_ids = str(history_str).split()
    vectors = [
        embeddings_dict[news_id]
        for news_id in history_ids
        if news_id in embeddings_dict
    ]

    if not vectors:
        return np.zeros(default_dim)

    return np.mean(vectors, axis=0)

def cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def rank_user_candidates(split="train"):
    """
    Ranks candidate impression news for each user based on vector cosine similarity.
    """
    behaviors_path = os.path.join(PROCESSED_DIR, f"behaviors_{split}.csv")
    if not os.path.exists(behaviors_path):
        print(f"File {behaviors_path} not found.")
        return

    embeddings = load_embeddings(split)
    df_behaviors = pd.read_csv(behaviors_path)

    print(f"\n--- Ranking Candidates for {split.upper()} split ---")

    for idx, row in df_behaviors.iterrows():
        impression_id = row["impression_id"]
        user_id = row["user_id"]
        history = row.get("history", "")
        impressions = str(row.get("impressions", ""))

        # 1. User Representation
        user_vec = compute_user_vector(history, embeddings)

        # 2. Score Candidate News Articles
        candidate_scores = []
        for candidate in impressions.split():
            parts = candidate.split("-")
            news_id = parts[0]
            label = parts[1] if len(parts) > 1 else None

            candidate_vec = embeddings.get(news_id, np.zeros(384))
            similarity_score = cosine_similarity(user_vec, candidate_vec)

            candidate_scores.append((news_id, similarity_score, label))

        # 3. Sort candidates by similarity score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\nUser: {user_id} (Impression ID: {impression_id})")
        print("Ranked Candidates (News ID | Score | Label):")
        for news_id, score, label in candidate_scores:
            lbl_str = f"| Label: {label}" if label else ""
            print(f"  -> {news_id} : Score = {score:.4f} {lbl_str}")

if __name__ == "__main__":
    rank_user_candidates("train")
    rank_user_candidates("dev")