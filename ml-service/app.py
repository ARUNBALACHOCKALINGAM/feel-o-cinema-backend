from flask import Flask, jsonify, request
import torch
from transformers import pipeline

app = Flask(__name__)

# Load Emotion Detection Model
emotion_model = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    device=0 if torch.cuda.is_available() else -1
)

@app.route("/recommend", methods=["POST"])
def get_movie_recommendation():
    mood = request.json.get("mood")
    detected_mood = detect_emotion(mood)
    movies = recommend_movies(detected_mood)
    return jsonify({"mood": detected_mood, "recommended_movies": movies})

if __name__ == "__main__":
    app.run(port=5001)