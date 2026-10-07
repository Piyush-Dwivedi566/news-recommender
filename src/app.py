import streamlit as st
import requests

st.set_page_config(page_title="News Recommendation System", layout="wide")

st.title("📰 News Recommendation Engine")
st.write("Real-time personalized news recommendations powered by vector similarity.")

# API Endpoint Configuration
API_URL = "http://127.0.0.1:8000/recommend"

st.sidebar.header("User Session Settings")

# Input fields for user history and candidate pool
history_input = st.sidebar.text_input(
    "User Reading History (Comma Separated)", 
    value="N10001, N10002"
)

candidates_input = st.sidebar.text_input(
    "Candidate Articles (Comma Separated)", 
    value="N10003, N10004, N10005"
)

top_k = st.sidebar.slider("Top Recommendations", min_value=1, max_value=5, value=3)

if st.sidebar.button("Get Recommendations"):
    history_ids = [x.strip() for x in history_input.split(",") if x.strip()]
    candidate_ids = [x.strip() for x in candidates_input.split(",") if x.strip()]

    payload = {
        "history_news_ids": history_ids,
        "candidate_news_ids": candidate_ids,
        "top_k": top_k
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            recommendations = response.json()
            
            st.subheader(f"Top {len(recommendations)} Recommendations")
            
            for rank, item in enumerate(recommendations, 1):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(label="Match Score", value=f"{item['score']:.4f}")
                with col2:
                    st.markdown(f"### {rank}. {item.get('title') or 'Article ' + item['news_id']}")
                    st.caption(f"**ID:** {item['news_id']} | **Category:** {item.get('category', 'N/A').upper()}")
                st.divider()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to API server at {API_URL}. Ensure uvicorn is running.")