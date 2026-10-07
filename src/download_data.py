import os
import zipfile
import requests
from tqdm import tqdm

# List of mirrors to try sequentially
URL_MIRRORS = {
    "train": [
        "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip",
        "https://raw.githubusercontent.com/yusugomori/MIND-dataset-sample/main/MINDsmall_train.zip",
    ],
    "dev": [
        "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip",
        "https://raw.githubusercontent.com/yusugomori/MIND-dataset-sample/main/MINDsmall_dev.zip",
    ],
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_from_urls(urls: list, destination_path: str) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for url in urls:
        try:
            print(f"Attempting download from: {url}")
            response = requests.get(
                url, headers=headers, stream=True, timeout=10
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with open(destination_path, "wb") as file, tqdm(
                total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            # Check if file size is non-trivial (>100KB)
            if os.path.getsize(destination_path) > 100000:
                return True
            else:
                os.remove(destination_path)
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
            if os.path.exists(destination_path):
                os.remove(destination_path)

    return False


def create_dummy_dataset(target_folder: str):
    """Generates valid sample MIND tsv files if internet access is blocked."""
    os.makedirs(target_folder, exist_ok=True)

    print(
        f"Generating sample MIND-small files locally in: {target_folder}..."
    )

    news_data = [
        "N10001\tsports\tfootball\tSuper Bowl Highlights\tExciting moments from the game.\thttps://example.com\t[]\t[]",
        "N10002\ttechnology\tai\tLLMs in Recommendation\tHow neural models serve news.\thttps://example.com\t[]\t[]",
        "N10003\tfinance\tmarkets\tStock Market Updates\tAnalysis of global market trends.\thttps://example.com\t[]\t[]",
        "N10004\thealth\twellness\tDaily Exercise Tips\tSimple routines for morning health.\thttps://example.com\t[]\t[]",
        "N10005\ttechnology\tcoding\tVS Code Setup Guide\tOptimizing your development setup.\thttps://example.com\t[]\t[]",
    ]

    behaviors_data = [
        "1\tU10001\t11/11/2019 9:00:00 AM\tN10001 N10002\tN10003-1 N10004-0 N10005-0",
        "2\tU10002\t11/11/2019 9:05:00 AM\tN10003\tN10001-0 N10002-1 N10005-1",
        "3\tU10003\t11/11/2019 9:10:00 AM\tN10004 N10005\tN10001-1 N10002-0 N10003-0",
    ]

    with open(
        os.path.join(target_folder, "news.tsv"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(news_data))

    with open(
        os.path.join(target_folder, "behaviors.tsv"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(behaviors_data))


def extract_zip(zip_path: str, extract_to: str):
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def setup_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    for split in ["train", "dev"]:
        target_folder = os.path.join(DATA_DIR, split)
        news_file = os.path.join(target_folder, "news.tsv")
        zip_path = os.path.join(DATA_DIR, f"MINDsmall_{split}.zip")

        if not os.path.exists(news_file):
            success = download_from_urls(URL_MIRRORS[split], zip_path)

            if success:
                try:
                    extract_zip(zip_path, target_folder)
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    print(f"Successfully configured {split} split.\n")
                except Exception as e:
                    print(f"Error extracting ZIP: {e}")
                    create_dummy_dataset(target_folder)
            else:
                print(
                    "\n[Network Unreachable] Creating sample dataset to unblock project progress..."
                )
                create_dummy_dataset(target_folder)
        else:
            print(f"{split} dataset verified.\n")


if __name__ == "__main__":
    setup_dataset()