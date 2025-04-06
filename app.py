from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import pickle
import gzip
from io import BytesIO
from surprise import SVD

app = Flask(__name__)
CORS(app)

# 🔗 Hugging Face base URL (raw file access)
HF_BASE_URL = "https://huggingface.co/vaishnaviii03/flickpick-models/resolve/main/"

# 📦 Load .pkl.gz from Hugging Face
def load_pickle_gz_from_hf(filename):
    url = HF_BASE_URL + filename
    response = requests.get(url)
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
        return pickle.load(f)

# 🔁 Load all resources
indices = load_pickle_gz_from_hf("indices.pkl.gz")
id_map = load_pickle_gz_from_hf("id_map.pkl.gz")
cosine_sim = load_pickle_gz_from_hf("cosine_sim.pkl.gz")
algo = load_pickle_gz_from_hf("algo.pkl.gz")
smd = load_pickle_gz_from_hf("smd_mini.pkl.gz")
indices_map = load_pickle_gz_from_hf("indices_map.pkl.gz")

# 🎞️ Poster fetch from TMDB
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        path = data.get('poster_path')
        if path:
            return f"https://image.tmdb.org/t/p/w500/{path}"
    except:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Image"

# 💡 Hybrid recommender
def hybrid(userId, title):
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame(columns=['title', 'movieId'])

    sim_scores = sorted(list(enumerate(cosine_sim[int(idx)])), key=lambda x: x[1], reverse=True)[1:16]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'movieId']].copy()

    movies['est'] = movies['movieId'].apply(lambda x: algo.predict(userId, indices_map.get(x, 'id')).est)
    return movies.sort_values('est', ascending=False).head(5)

# ✅ Health check route
@app.route('/')
def home():
    return "🎬 FlickPick API is running with Hugging Face models"

# 🔍 Recommendation route
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        userId = int(data['userId'])
        movie = data['movie']

        recommendations = hybrid(userId, movie)

        result = []
        for _, row in recommendations.iterrows():
            result.append({
                'title': row['title'],
                'movieId': row['movieId'],
                'poster': fetch_poster(row['movieId'])
            })

        return jsonify({'recommendations': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
