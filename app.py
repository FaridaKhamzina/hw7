import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.1.0")

app = FastAPI(title="HW7 ML Service", version=MODEL_VERSION)


class PredictRequest(BaseModel):
    x: List[float] = Field(..., description="Numeric feature vector")


@app.get("/health")
def health() -> dict:
    """Health-check endpoint for CI/CD and load balancer checks."""
    return {"status": "ok", "version": MODEL_VERSION}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    """A minimal deterministic ML-like prediction endpoint.

    In a real project this function would load a trained model artifact.
    Here we keep the service intentionally small for CI/CD practice.
    """
    if not request.x:
        raise HTTPException(status_code=400, detail="Input vector x must not be empty")

    score = float(sum(request.x))

    # Version v1.1.0 uses a slightly different threshold to emulate a new model.
    threshold = 5.0 if MODEL_VERSION == "v1.1.0" else 6.0
    prediction = int(score >= threshold)

    return {
        "prediction": prediction,
        "score": score,
        "threshold": threshold,
        "version": MODEL_VERSION,
    }
