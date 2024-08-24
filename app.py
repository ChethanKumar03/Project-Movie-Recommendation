from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your movie dataset
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')  # Modify this based on your actual file structure

movies = movies.merge(credits, left_on='title', right_on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies = movies[['movie_id', 'title', 'overview', 'tags']]

movies['tags'] = movies['tags'].apply(lambda x: x.lower())

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to calculate top 10 movies
def get_top_10_movies(user_input, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == user_input].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_movies = []
    movie_titles = movies['title'].tolist()
    if request.method == 'POST':
        user_input = request.form.get('movie_select')
        selected_movies = get_top_10_movies(user_input)
    return render_template('index.html', movies=selected_movies, movie_titles=movie_titles)

if __name__ == '__main__':
    app.run(debug=True)
