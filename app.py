from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import requests
from surprise import SVD

app = Flask(__name__)
CORS(app)

# 📦 Load models and data
indices = pickle.load(open('model/indices.pkl', 'rb'))
id_map = pickle.load(open('model/id_map.pkl', 'rb'))
cosine_sim = pickle.load(open('model/cosine_sim.pkl', 'rb'))
algo = pickle.load(open('model/algo.pkl', 'rb'))
smd = pickle.load(open('model/smd.pkl', 'rb'))
indices_map = pickle.load(open('model/indices_map.pkl', 'rb'))

# 📸 Fetch poster from TMDB
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Image"

# 🔀 Hybrid recommendation engine
def hybrid(userId, title, indices, cosine_sim, algo, smd, indices_map):
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame(columns=['title', 'movieId'])
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'movieId']]
    
    # Estimate rating for user
    movies['est'] = movies['movieId'].apply(lambda x: algo.predict(userId, indices_map.get(x, 'id')).est)
    movies = movies.sort_values('est', ascending=False)
    
    return movies.head(10)

# ✅ Health check
@app.route('/')
def home():
    return "🎬 FlickPick API is running"

# 🔍 POST API endpoint
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        userId = int(data['userId'])
        movie = data['movie']
        recommendations = hybrid(userId, movie, indices, cosine_sim, algo, smd, indices_map)

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

if __name__ == '__main__':
    app.run(debug=True)
