from flask import Flask, jsonify, request
import torch
from transformers import pipeline
from datasets import load_dataset
import os
import gc

app = Flask(__name__)

# Configuration
MAX_MOVIES = 5  # Reduced from 10
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

# Optimize memory settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Memory-efficient loading
def load_components():
    """Lazy loading of heavy resources"""
    global emotion_model, dataset
    
    # 1. Load model with memory optimizations
    emotion_model = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1,  # Force CPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # 2. Stream dataset instead of loading all at once
    dataset = load_dataset("wykonos/movies", streaming=True)
    
    # 3. Immediate garbage collection
    gc.collect()

# Load components at first request
@app.before_first_request
def initialize():
    load_components()

# Optimized recommendation logic
def recommend_movies(mood):
    emotion_to_genre = {
        # ... (keep your existing mapping) ...
    }
    
    recommended_genres = emotion_to_genre.get(mood, ["Comedy"])
    
    # Stream dataset directly without pandas
    movies = []
    for movie in dataset["train"]:
        if any(genre in movie["genres"] for genre in recommended_genres):
            movies.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "poster_path": movie.get("poster_path", ""),
                "vote_average": movie.get("vote_average", 0),
                "release_date": movie.get("release_date", "")
            })
            if len(movies) >= MAX_MOVIES:
                break
    
    return movies


# Detect Emotion
def detect_emotion(user_input):
    emotions = emotion_model(user_input)
    detected_emotion = max(emotions, key=lambda e: e["score"])["label"]
    return detected_emotion

@app.route("/recommend", methods=["POST"])
def get_movie_recommendation():
    mood = request.json.get("mood")
    detected_mood = detect_emotion(mood)
    movies = recommend_movies(detected_mood)
    return jsonify({"mood": detected_mood, "recommended_movies": movies})

if __name__ == "__main__":
    app.run(port=5001)