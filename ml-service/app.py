from flask import Flask, jsonify, request
import torch
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import os
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MAX_MOVIES = 3  # Reduced from 5
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
DATASET_NAME = "wykonos/movies"

# Optimize memory settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

# Memory-efficient components
def load_model():
    """Load model with strict memory optimizations"""
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
        device=-1,  # Force CPU
        torch_dtype=torch.float16,
        framework="pt",
        truncation=True,
        max_length=128  # Reduced sequence length
    )

def stream_dataset():
    """Generator for streaming dataset"""
    for movie in load_dataset(DATASET_NAME, streaming=True)["train"]:
        yield movie
        del movie  # Explicit memory release
        gc.collect()

# Lazy-loaded components
emotion_model = None
movie_stream = None

@app.before_first_request
def initialize():
    global emotion_model, movie_stream
    logger.info("Initializing resources...")
    
    # Load model first
    emotion_model = load_model()
    
    # Initialize dataset stream
    movie_stream = stream_dataset()
    
    logger.info("Resource initialization complete")

# Optimized recommendation logic
def recommend_movies(mood):
    emotion_to_genre = {
        # ... (keep your existing mapping) ...
    }
    
    recommended_genres = emotion_to_genre.get(mood, ["Comedy"])
    movies = []
    
    try:
        for movie in movie_stream:
            if any(genre in movie.get("genres", "") for genre in recommended_genres):
                movies.append({
                    "title": movie.get("title", ""),
                    "genres": movie.get("genres", ""),
                    "poster_path": movie.get("poster_path", ""),
                })
                if len(movies) >= MAX_MOVIES:
                    break
            del movie
            gc.collect()
    except Exception as e:
        logger.error(f"Dataset streaming error: {str(e)}")
    
    return movies


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