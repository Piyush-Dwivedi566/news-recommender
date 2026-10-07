import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_embeddings(split="train"):
    path = os.path.join(PROCESSED_DIR, f"embeddings_{split}.npy")
    return np.load(path, allow_pickle=True).item()

def compute_user_vector(history_str, embeddings_dict, default_dim=384):
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
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def calculate_mrr(y_true, y_score):
    """Calculates Mean Reciprocal Rank for an impression."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    ranks = np.where(y_true_sorted == 1)[0]
    if len(ranks) == 0:
        return 0.0
    return 1.0 / (ranks[0] + 1)

def calculate_ndcg(y_true, y_score, k=10):
    """Calculates Normalized Discounted Cumulative Gain at K."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)[:k]
    
    # DCG
    gains = 2**y_true_sorted - 1
    discounts = np.log2(np.arange(2, len(y_true_sorted) + 2))
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal_y_true = np.sort(y_true)[::-1][:k]
    ideal_gains = 2**ideal_y_true - 1
    ideal_discounts = np.log2(np.arange(2, len(ideal_y_true) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_model(split="dev"):
    behaviors_path = os.path.join(PROCESSED_DIR, f"behaviors_{split}.csv")
    if not os.path.exists(behaviors_path):
        print(f"File {behaviors_path} not found.")
        return

    embeddings = load_embeddings(split)
    df_behaviors = pd.read_csv(behaviors_path)

    auc_list, mrr_list, ndcg5_list, ndcg10_list = [], [], [], []

    for idx, row in df_behaviors.iterrows():
        history = row.get("history", "")
        impressions = str(row.get("impressions", ""))

        user_vec = compute_user_vector(history, embeddings)

        y_true = []
        y_score = []

        for candidate in impressions.split():
            parts = candidate.split("-")
            news_id = parts[0]
            label = int(parts[1]) if len(parts) > 1 else 0

            candidate_vec = embeddings.get(news_id, np.zeros(384))
            score = cosine_similarity(user_vec, candidate_vec)

            y_true.append(label)
            y_score.append(score)

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Skip impressions without positive and negative instances for AUC calculation
        if len(np.unique(y_true)) > 1:
            auc_list.append(roc_auc_score(y_true, y_score))
            mrr_list.append(calculate_mrr(y_true, y_score))
            ndcg5_list.append(calculate_ndcg(y_true, y_score, k=5))
            ndcg10_list.append(calculate_ndcg(y_true, y_score, k=10))

    print(f"\n================ EVALUATION RESULTS ({split.upper()}) ================")
    print(f"AUC     : {np.mean(auc_list):.4f}")
    print(f"MRR     : {np.mean(mrr_list):.4f}")
    print(f"nDCG@5  : {np.mean(ndcg5_list):.4f}")
    print(f"nDCG@10 : {np.mean(ndcg10_list):.4f}")
    print("==========================================================")

if __name__ == "__main__":
    evaluate_model("dev")