# app/main.py
from fastapi import FastAPI, Request
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from google.cloud import aiplatform
# from starlette.concurrency import run_in_threadpool
import os
import requests

app = FastAPI()

# --- Environment Variables for GCP ---
# It's a good practice to get these from environment variables.
# You would set these in your Cloud Run service configuration or your local environment.
# GCP_PROJECT = os.getenv("GCP_PROJECT")
# GCP_LOCATION = os.getenv("GCP_LOCATION")  # e.g., "us-central1"
# ENDPOINT_ID = os.getenv("ENDPOINT_ID")

# --- FastAPI Events ---
# @app.on_event("startup")
# async def startup_event():
#     """
#     On startup, validate environment variables and initialize the Vertex AI client.
#     Note: @app.on_event is deprecated. For newer FastAPI versions, prefer using
#     the 'lifespan' context manager.
#     """
#     if not all([GCP_PROJECT, GCP_LOCATION, ENDPOINT_ID]):
#         raise RuntimeError("GCP_PROJECT, GCP_LOCATION, and ENDPOINT_ID environment variables must be set.")
#     aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)

# Allow CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Angular app URL in production
    # In production, you should restrict this to your frontend's domain
    # For example: allow_origins=["https://your-angular-app.com", "https://www.your-angular-app.com"]
    # Using ["*"] is a security risk.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Message(BaseModel):
    message: str

class PredictionInput(BaseModel):
    # This model should be adapted to match the expected input of your Vertex AI model
    prompt: str
    steps: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the FastAPI middleware!"}

@app.post("/api/message")
async def receive_message(msg: Message):
    return {"reply": f"Received your message: {msg.message}"}

@app.post("/api/predict/forecast")
async def predict(data: PredictionInput):
    """
    Receives input data and returns a prediction from a Vertex AI model endpoint.
    """
    # url = "https://my-python-app-888916268766.asia-south1.run.app/forecast"
    url = "https://arima-forecast-api1-888916268766.asia-south1.run.app/forecast"
    try:
        response = requests.post(url, json = data.dict(), timeout=60)
        # print(f"Status Code: {response.status_code}")
        # print(f"Response: {response.json()}")
        return {"prediction": response.json()}
    except requests.exceptions.RequestException as e:
        # Log the error for debugging purposes
        print(f"An error occurred during prediction request: {e}")
        # Return a more specific error response to the client
        raise HTTPException(status_code=502, detail="Error communicating with the prediction service.")
    except Exception as e:
        # Log the error for debugging purposes
        print(f"An error occurred during prediction: {e}")
        print(f"An unexpected error occurred during prediction: {e}")
        # Return a generic error response to the client
        raise HTTPException(status_code=500, detail="An error occurred while making a prediction.")


# DEPLOY COMMAND
# gcloud run deploy fastapi-middleware \
#   --project your-gcp-project-id \
#   --image asia-south2-docker.pkg.dev/dotengage/fastapi-repo/fastapi-middleware \
#   --platform managed \
#   --region asia-south2 \
#   --allow-unauthenticated \
#   --set-env-vars GCP_PROJECT=your-gcp-project-id,GCP_LOCATION=your-gcp-location,ENDPOINT_ID=your-endpoint-id
