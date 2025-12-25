from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv
from recommender import recommend_movies

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "/static/no_poster.jpg"

app = Flask(__name__)

def fetch_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }
        r = requests.get(url, params=params, timeout=5)

        if r.status_code == 200:
            data = r.json()
            if data["results"]:
                poster = data["results"][0].get("poster_path")
                if poster:
                    return POSTER_BASE_URL + poster

        return PLACEHOLDER

    except Exception as e:
        print("TMDB error:", e)
        return PLACEHOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form.get("movie_name", "").strip()

    indices = recommend_movies(movie_name)

    if not indices:
        movies = [{
            "title": "Movie not found",
            "poster": PLACEHOLDER
        }]
    else:
        movies = []
        for movie in indices:
            title = movie["title"]
            movies.append({
    "title": movie["title"],
    "poster": fetch_poster(movie["title"]) or "/static/no_poster.jpg",
    "reason": movie["reason"]
})



    return render_template(
        "recommend.html",
        movie_name=movie_name,
        movies=movies
    )


if __name__ == "__main__":
    print("🎬 Movie Recommender running with TMDB posters")
    app.run(debug=True)
