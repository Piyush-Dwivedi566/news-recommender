import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI(title="News Recommendation API", version="1.0")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Global cache for embeddings and news metadata
embeddings_cache = {}
news_metadata_cache = {}


class RecommendationRequest(BaseModel):
    history_news_ids: List[str]
    candidate_news_ids: List[str]
    top_k: Optional[int] = 5


class RecommendedArticle(BaseModel):
    news_id: str
    score: float
    category: Optional[str] = None
    title: Optional[str] = None


@app.on_event("startup")
def load_resources():
    global embeddings_cache, news_metadata_cache
    
    train_path = os.path.join(PROCESSED_DIR, "embeddings_train.npy")
    dev_path = os.path.join(PROCESSED_DIR, "embeddings_dev.npy")

    if os.path.exists(train_path):
        embeddings_cache.update(np.load(train_path, allow_pickle=True).item())
    if os.path.exists(dev_path):
        embeddings_cache.update(np.load(dev_path, allow_pickle=True).item())

    for split in ["train", "dev"]:
        csv_path = os.path.join(PROCESSED_DIR, f"news_{split}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                news_metadata_cache[row["news_id"]] = {
                    "category": str(row.get("category", "")),
                    "title": str(row.get("title", "")),
                }


def compute_user_vector(history_ids: List[str], default_dim: int = 384):
    vectors = [
        embeddings_cache[news_id]
        for news_id in history_ids
        if news_id in embeddings_cache
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


@app.get("/")
def root():
    return {"status": "online", "message": "News Recommender API running."}


@app.post("/recommend", response_model=List[RecommendedArticle])
def recommend(request: RecommendationRequest):
    if not request.candidate_news_ids:
        raise HTTPException(status_code=400, detail="Candidate news IDs list cannot be empty.")

    user_vec = compute_user_vector(request.history_news_ids)
    scored_candidates = []

    for news_id in request.candidate_news_ids:
        cand_vec = embeddings_cache.get(news_id, np.zeros(384))
        score = cosine_similarity(user_vec, cand_vec)
        meta = news_metadata_cache.get(news_id, {})

        scored_candidates.append(
            RecommendedArticle(
                news_id=news_id,
                score=round(score, 4),
                category=meta.get("category"),
                title=meta.get("title"),
            )
        )

    scored_candidates.sort(key=lambda x: x.score, reverse=True)
    return scored_candidates[: request.top_k]