from flask import Flask, jsonify, request
import torch
from transformers import pipeline
from datasets import load_dataset

app = Flask(__name__)

# Load Emotion Detection Model
emotion_model = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    device=0 if torch.cuda.is_available() else -1
)

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

# Recommend Movies
def recommend_movies(mood):
    # Get recommended genres based on mood
    recommended_genres = emotion_to_genre.get(mood, ["Comedy"])
    
    # Filter movies that match the recommended genres
    filtered_movies = df[df["genres"].apply(
        lambda g: isinstance(g, str) and any(genre in g for genre in recommended_genres)
    )]
    
    # Check if there are enough movies to sample
    if len(filtered_movies) < 10:
        # If fewer than 10 movies, return all available movies
        movie_list = filtered_movies[["title", "genres", "poster_path", "vote_average", "release_date"]].dropna().to_dict(orient="records")
    else:
        # Randomly sample 10 movies
        movie_list = filtered_movies[["title", "genres", "poster_path", "vote_average", "release_date"]].dropna().sample(n=10).to_dict(orient="records")
    
    return movie_list

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