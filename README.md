# News Recommender System

A lightweight, end-to-end news recommendation pipeline built using Python, FastAPI, and Streamlit. This project processes article metadata and user interaction logs, generates semantic vector embeddings using SentenceTransformers, ranks candidate articles based on user interest profiles, and serves predictions through a REST API and an interactive web interface.


## What This Project Does

When users browse news platforms, recommending relevant articles based on past reading habits keeps them engaged. 

This project implements a content-based vector similarity approach:
1. **Article Vectors:** News headlines and categories are converted into 384-dimensional dense embeddings using `all-MiniLM-L6-v2`.
2. **User Profiles:** A user's profile is calculated by averaging the embeddings of articles they've previously read.
3. **Similarity Ranking:** Candidate news items are scored using cosine similarity against the user vector and returned in ranked order.
4. **API & UI Layer:** The ranker runs behind a FastAPI backend and is visualized using a Streamlit frontend.


## Project Architecture

```text
               +-----------------------+
               |  MIND Dataset (CSVs)  |
               +-----------+-----------+
                           |
                           v
              +-------------------------+
              | Data Parsing & Loader   |
              +------------+------------+
                           |
                           v
         +----------------------------------+
         | Sentence Transformers Embedder   |
         |      (all-MiniLM-L6-v2)          |
         +-----------------+----------------+
                           |
                           v
           +-------------------------------+
           |  Ranker Engine & Vector Space |
           |  (Aggregated Cosine Sim Score)|
           +---------------+---------------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
  +-------------------+       +-----------------------+
  |  Offline Metrics  |       |   FastAPI REST App    |
  | (AUC, MRR, nDCG)  |       |   (port 8000 /docs)   |
  +-------------------+       +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Streamlit Dashboard   |
                              |      (port 8501)      |
                              +-----------------------+

# Tech Stack & Key Libraries
Language: Python 3.12

Embeddings & ML: sentence-transformers, torch (CPU), scikit-learn, numpy, pandas

API Backend: fastapi, uvicorn

Frontend: streamlit

# Quickstart Guide
1. Environment Setup
Clone the repository and set up a virtual environment:

DOS
git clone [https://github.com/YOUR_USERNAME/news-recommender.git](https://github.com/YOUR_USERNAME/news-recommender.git)
cd news-recommender
python -m venv venv
venv\Scripts\activate
2. Install Dependencies
Install CPU-only PyTorch followed by the rest of the requirements:

DOS
pip install --default-timeout=100 torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install pandas numpy sentence-transformers scikit-learn fastapi uvicorn streamlit requests

3. Generate Embeddings & Evaluate
Run the preprocessing and ranking pipeline:

DOS
python src/data_loader.py
python src/preprocess.py
python src/embedder.py
python src/evaluator.py


# Running the Application
To interact with the app, launch both the API backend and Streamlit UI in parallel terminal windows.

Terminal 1: FastAPI Backend
DOS
uvicorn src.api:app --reload --port 8000
API Documentation / Swagger UI: http://127.0.0.1:8000/docs

Terminal 2: Streamlit Dashboard

DOS
streamlit run src/app.py
Web Interface: http://localhost:8501

# Metric Results
Evaluated on user click behavioral sessions:

MRR: ~0.6667

nDCG@5 / nDCG@10: ~0.7748

AUC: 0.5000 (Baseline on sample sets)

# License
Distributed under the MIT License. Feel free to use and adapt!