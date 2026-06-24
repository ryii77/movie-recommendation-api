from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import json


# Define the same model structure used in the notebook
class MovieRecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)

        x = torch.cat(
            [user_emb, movie_emb],
            dim=1
        )

        logits = self.fc(x).squeeze(1)

        return logits


# Create the FastAPI application
app = FastAPI(
    title="Movie Recommendation Microservice",
    description=(
        "This service predicts whether a user is likely "
        "to like a specific movie."
    ),
    version="1.0"
)


# Load the saved model file
checkpoint = torch.load(
    "movie_recommendation_model.pth",
    map_location=torch.device("cpu")
)


# Recreate the model
model = MovieRecommendationModel(
    num_users=checkpoint["num_users"],
    num_movies=checkpoint["num_movies"],
    embedding_dim=checkpoint["embedding_dim"]
)


# Load the trained model parameters
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# Load user and movie ID mappings
with open(
    "id_mappings.json",
    "r",
    encoding="utf-8"
) as file:
    mappings = json.load(file)


user_mapping = mappings["user_mapping"]
movie_mapping = mappings["movie_mapping"]


# Load movie titles
with open(
    "movie_titles.json",
    "r",
    encoding="utf-8"
) as file:
    movie_titles = json.load(file)


# Define the expected input format
class PredictionInput(BaseModel):
    user_id: int
    movie_id: int


# Home page
@app.get("/")
def home():
    return {
        "message": "Movie recommendation service is running.",
        "instructions": "Open /docs to test the service."
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    user_key = str(data.user_id)
    movie_key = str(data.movie_id)

    # Check whether the user exists
    if user_key not in user_mapping:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown user_id: {data.user_id}. "
                "Please enter a user ID contained in "
                "the MovieLens dataset."
            )
        )

    # Check whether the movie exists
    if movie_key not in movie_mapping:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown movie_id: {data.movie_id}. "
                "Please enter a movie ID contained in "
                "the MovieLens dataset."
            )
        )

    # Convert original IDs to encoded model indices
    user_index = user_mapping[user_key]
    movie_index = movie_mapping[movie_key]

    user_tensor = torch.tensor(
        [user_index],
        dtype=torch.long
    )

    movie_tensor = torch.tensor(
        [movie_index],
        dtype=torch.long
    )

    # Run the model
    with torch.no_grad():
        logit = model(
            user_tensor,
            movie_tensor
        )

        probability = torch.sigmoid(logit).item()

    # Convert probability into a binary prediction
    prediction = int(probability >= 0.5)

    if prediction == 1:
        recommendation = (
            "The user is likely to like this movie."
        )
    else:
        recommendation = (
            "The user is not likely to like this movie."
        )

    return {
        "user_id": data.user_id,
        "movie_id": data.movie_id,
        "movie_title": movie_titles.get(
            movie_key,
            "Unknown title"
        ),
        "probability": round(probability, 4),
        "prediction": prediction,
        "recommendation": recommendation
    }