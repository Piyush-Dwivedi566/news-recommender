import os
import pandas as pd


def load_news_data(file_path: str) -> pd.DataFrame:
    """Reads news.tsv into a clean pandas DataFrame."""
    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    df = pd.read_csv(file_path, sep="\t", names=columns)
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    return df


def load_behaviors_data(file_path: str) -> pd.DataFrame:
    """Reads behaviors.tsv into a pandas DataFrame."""
    columns = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]
    df = pd.read_csv(file_path, sep="\t", names=columns)
    df["history"] = df["history"].fillna("")
    return df


def process_dataset(raw_dir: str, processed_dir: str):
    """Parses raw files for train/dev and saves processed files."""
    os.makedirs(processed_dir, exist_ok=True)

    for split in ["train", "dev"]:
        split_path = os.path.join(raw_dir, split)
        news_file = os.path.join(split_path, "news.tsv")
        behaviors_file = os.path.join(split_path, "behaviors.tsv")

        if not os.path.exists(news_file) or not os.path.exists(behaviors_file):
            print(f"Skipping {split}: files not found in {split_path}")
            continue

        print(f"\n--- Processing {split.upper()} split ---")
        df_news = load_news_data(news_file)
        df_behaviors = load_behaviors_data(behaviors_file)

        print(f"Loaded {len(df_news):,} articles.")
        print(f"Loaded {len(df_behaviors):,} user impression sessions.")

        news_out = os.path.join(processed_dir, f"news_{split}.csv")
        behaviors_out = os.path.join(processed_dir, f"behaviors_{split}.csv")

        # Save to CSV to prevent parquet overwrite/engine issues
        df_news.to_csv(news_out, index=False)
        df_behaviors.to_csv(behaviors_out, index=False)

        print(f"Saved: {news_out}")
        print(f"Saved: {behaviors_out}")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    process_dataset(raw_dir, processed_dir)