import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def generate_news_embeddings(split="train", model_name="all-MiniLM-L6-v2"):
    """Generates dense vector embeddings for news titles and abstracts."""
    from sentence_transformers import SentenceTransformer

    file_path = os.path.join(PROCESSED_DIR, f"news_{split}.csv")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please run parse_data.py first.")
        return

    print(f"Loading processed news data from {file_path}...")
    df_news = pd.read_csv(file_path)

    # Combine Category, Title, and Abstract for rich representation
    df_news["combined_text"] = (
        df_news["category"].fillna("") + " " +
        df_news["title"].fillna("") + ": " +
        df_news["abstract"].fillna("")
    )

    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(df_news)} news articles...")
    texts = df_news["combined_text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Save embeddings mapping dictionary
    news_ids = df_news["news_id"].tolist()
    embedding_dict = dict(zip(news_ids, embeddings))

    output_path = os.path.join(PROCESSED_DIR, f"embeddings_{split}.npy")
    np.save(output_path, embedding_dict, allow_pickle=True)
    print(f"Successfully saved embeddings to: {output_path} (Vector dimension: {embeddings.shape[1]})")

if __name__ == "__main__":
    try:
        generate_news_embeddings("train")
        generate_news_embeddings("dev")
    except ModuleNotFoundError:
        print("Installing missing 'sentence-transformers' library...")
        os.system("pip install sentence-transformers")
        generate_news_embeddings("train")
        generate_news_embeddings("dev")