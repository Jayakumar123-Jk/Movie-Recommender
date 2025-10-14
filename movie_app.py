
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- Paths ----
BASE_DIR = os.path.dirname(__file__)
MOVIES_CSV = os.path.join(BASE_DIR, "movies.csv")

# ---- Helpers ----
@st.cache_data
def load_movies(path):
    """Load movies CSV into a DataFrame. Must have at least 'title' column.
       Common schemas: movieId,title,genres
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"movies.csv not found at: {path}")
    df = pd.read_csv(path)
    # Basic normalization: ensure needed columns
    if 'title' not in df.columns:
        raise ValueError("movies.csv must contain a 'title' column.")
    # create a text field to vectorize: title + genres (if present)
    if 'genres' in df.columns:
        df['text'] = df['title'].astype(str) + " " + df['genres'].astype(str)
    else:
        df['text'] = df['title'].astype(str)
    # Reset index and keep a mapping for titles
    df = df.reset_index(drop=True)
    return df

@st.cache_resource
def build_similarity_matrix(texts):
    """Build TF-IDF matrix and return cosine similarity matrix"""
    # small preprocessing
    tf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)
    tfidf_matrix = tf.fit_transform(texts)
    # linear_kernel gives cosine similarity for TF-IDF
    sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    return sim_matrix, tf

def get_recommendations(title, df, sim_matrix, top_n=10):
    """Return top_n recommended movie titles for a given title"""
    # find index of the movie
    try:
        idx = df.index[df['title'] == title][0]
    except IndexError:
        # fallback: try contains or case-insensitive match
        matches = df[df['title'].str.lower().str.contains(title.lower())]
        if len(matches) > 0:
            idx = matches.index[0]
        else:
            return []
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # skip the first (itself)
    top_indices = [i for i, score in sim_scores[1: top_n + 1]]
    return df.iloc[top_indices].copy()

# ---- App UI ----
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender (Streamlit)")
st.markdown(
    "This simple content-based recommender uses TF-IDF on **movie title + genres** "
    "to find similar movies. No pickle required — just `movies.csv`."
)

# Load data
try:
    movies_df = load_movies(MOVIES_CSV)
except Exception as e:
    st.error(f"Error loading movies.csv: {e}")
    st.stop()

# Build similarity matrix
with st.spinner("Building recommendation engine (TF-IDF). This happens once..."):
    sim_matrix, tf_vectorizer = build_similarity_matrix(movies_df['text'])

# Sidebar: options
st.sidebar.header("Options")
search_text = st.sidebar.text_input("Search movie title (type or paste)", "")
top_k = st.sidebar.slider("How many recommendations?", 1, 20, 10)
show_scores = st.sidebar.checkbox("Show similarity scores", value=False)

# Main: searchable selectbox
st.markdown("### Pick a movie to get recommendations")
# If user typed text, attempt to narrow the selectable list
if search_text:
    candidates = movies_df[movies_df['title'].str.contains(search_text, case=False, na=False)]
    if len(candidates) == 0:
        st.warning("No exact matches found for search. Showing full list.")
        movie_list = movies_df['title'].tolist()
    else:
        movie_list = candidates['title'].tolist()
else:
    movie_list = movies_df['title'].tolist()

selected = st.selectbox("Select movie", movie_list)

if st.button("Get recommendations"):
    if not selected:
        st.warning("Please select a movie.")
    else:
        with st.spinner("Computing recommendations..."):
            rec_df = get_recommendations(selected, movies_df, sim_matrix, top_n=top_k)
        if rec_df is None or rec_df.empty:
            st.info("No recommendations found.")
        else:
            # Prepare display
            display_df = rec_df[['title']].copy()
            if 'genres' in rec_df.columns:
                display_df['genres'] = rec_df['genres']
            if show_scores:
                # compute scores against selected
                base_idx = movies_df.index[movies_df['title'] == selected][0]
                scores = [sim_matrix[base_idx][i] for i in rec_df.index]
                display_df['score'] = np.round(scores, 4)
            st.subheader(f"Top {len(display_df)} recommendations for: **{selected}**")
            st.table(display_df.reset_index(drop=True))

# Footer: small helpful tips
st.markdown("---")
st.markdown(
    "Tips: If your CSV uses different column names, open `movie_app.py` and ensure it has a `title` column. "
    "If your dataset is large, consider precomputing embeddings offline and saving them as a pickle (not required)."
)
