from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Initialize FastAPI application once with metadata
app = FastAPI(title="Medical Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load pre-trained machine learning models
health_model = joblib.load(MODEL_DIR / "health_condition_model.pkl")
brain_model = joblib.load(MODEL_DIR / "brain_tumor_detection_model.pkl")


@app.get("/")
def home():
    return {
        "message": "Medical Prediction API is running successfully",
    }


class HealthInput(BaseModel):
    features: list


def _prepare_health_features(features):
    if not isinstance(features, list) or len(features) != 5:
        raise ValueError("Expected a list of 5 features: [age_years, gender, heart_rate_bpm, spo2_percent, body_temperature_c]")

    age_years, gender, heart_rate_bpm, spo2_percent, body_temperature_c = features

    if isinstance(gender, str):
        gender_value = gender.strip()
        if gender_value.lower() == "male":
            gender_value = "Male"
        elif gender_value.lower() == "female":
            gender_value = "Female"
        else:
            raise ValueError("Gender must be 'Male' or 'Female'")
    else:
        gender_value = str(gender)

    return pd.DataFrame([
        {
            "age_years": float(age_years),
            "gender": gender_value,
            "heart_rate_bpm": float(heart_rate_bpm),
            "spo2_percent": float(spo2_percent),
            "body_temperature_c": float(body_temperature_c),
        }
    ])


@app.post("/health")
def predict_health(data: HealthInput):
    try:
        x = _prepare_health_features(data.features)
        prediction = health_model.predict(x)
        return {
            "prediction": str(prediction[0])
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/brain")
async def predict_brain(file: UploadFile = File(...)):
    # Open and process the uploaded image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))

    # Normalize and reshape to match model input requirements (1, 224, 224, 3)
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)

    prediction = brain_model.predict(image_array)

    return {
        "prediction": prediction.tolist()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
