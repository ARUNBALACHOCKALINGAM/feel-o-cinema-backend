from flask import Flask, jsonify, request
from datasets import load_dataset
import requests
import os
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hugging Face Space URL for Emotion Detection
EMOTION_API_URL = "https://arunbala21-feel-o-cinema.hf.space/run/predict"

# Load Movie Dataset
dataset = load_dataset("wykonos/movies")
df = dataset["train"].to_pandas()

# Emotion-to-Genre Mapping
emotion_to_genre = {
    "sadness": ["Feel-good", "Comedy", "Family", "Adventure"],
    "joy": ["Romance", "Drama", "Thriller"],
    "fear": ["Comedy", "Fantasy", "Lighthearted Action"],
    "anger": ["Comedy", "Inspirational", "Sports"],
    "neutral": ["Sci-Fi", "Mystery", "Thriller"],
    "tired": ["Documentary", "Slow-burn Drama"],
    "love": ["Romance", "Musical", "Comedy"]
}

# Detect Emotion
def detect_emotion(user_input):
    try:
        response = requests.post(
            EMOTION_API_URL,
            json={"data": [user_input]},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            detected_mood = result["data"][0]  # Extract mood from response
            return detected_mood
        else:
            logger.error(f"Error calling emotion API: {response.status_code} - {response.text}")
            return "neutral"  # Default fallback mood
    except Exception as e:
        logger.error(f"Exception while calling emotion API: {str(e)}")
        return "neutral"

# Recommend Movies
def recommend_movies(mood):
    recommended_genres = emotion_to_genre.get(mood, ["Comedy"])

    filtered_movies = df[df["genres"].apply(
        lambda g: isinstance(g, str) and any(genre in g for genre in recommended_genres)
    )]

    if len(filtered_movies) < 10:
        movie_list = filtered_movies[["title", "genres", "poster_path", "vote_average", "release_date"]].dropna().to_dict(orient="records")
    else:
        movie_list = filtered_movies[["title", "genres", "poster_path", "vote_average", "release_date"]].dropna().sample(n=10).to_dict(orient="records")

    return movie_list

@app.route("/recommend", methods=["POST"])
def get_movie_recommendation():
    data = request.json
    user_input = data.get("mood", "")

    if not user_input:
        return jsonify({"error": "No mood provided"}), 400

    detected_mood = detect_emotion(user_input)
    movies = recommend_movies(detected_mood)

    return jsonify({"mood": detected_mood, "recommended_movies": movies})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
