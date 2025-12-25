import pandas as pd
import ast
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt")
nltk.download("stopwords")


df = pd.read_csv("tmdb_5000_movies.csv")


stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


df["combined"] = (
    df["genres"].fillna("") + " " +
    df["keywords"].fillna("") + " " +
    df["overview"].fillna("")
)

df["cleaned"] = df["combined"].apply(clean_text)


tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["cleaned"])
cosine_sim = cosine_similarity(tfidf_matrix)


def extract_genres(genre_str):
    """
    Extract genre names from TMDB JSON-style string
    """
    try:
        genres = ast.literal_eval(genre_str)
        return set(g["name"].lower() for g in genres)
    except:
        return set()


def recommend_movies(movie_name, top_n=5):
    """
    Returns a list of dicts:
    [
      { "title": "...", "reason": "..." },
      ...
    ]
    """

    movie_name = movie_name.lower().strip()

  
    match = df[df["title"].str.lower() == movie_name]
    if match.empty:
        return []

    base_idx = match.index[0]
    base_genres = extract_genres(df.iloc[base_idx]["genres"])


    scores = list(enumerate(cosine_sim[base_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i, _ in scores[1:]:
        if len(results) == top_n:
            break

        title = df.iloc[i]["title"]
        rec_genres = extract_genres(df.iloc[i]["genres"])
        common = base_genres & rec_genres

        if common:
            reason = "Shares " + ", ".join(list(common)[:2]) + " genre(s)"
        else:
            reason = "Similar storyline and themes"

        results.append({
            "title": title,
            "reason": reason
        })

    return results
