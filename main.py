from io import BytesIO
from pathlib import Path
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# TensorFlow / Keras for the deep learning model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Initialize FastAPI application once with metadata
app = FastAPI(title="Medical Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"  # Ensure your models are inside a 'models' folder, or update BASE_DIR accordingly

# 1. Load pre-trained machine learning models & mappings
health_model = joblib.load(MODEL_DIR / "health_condition_model.pkl")

# Load your Keras brain tumor model (.keras format)
brain_model = load_model(MODEL_DIR / "brain_tumor_detection_model.keras")

# Load the class mapping JSON to convert numbers back to labels
with open(MODEL_DIR / "class_mapping.json", "r") as f:
    class_indices = json.load(f)

# Invert dictionary: e.g., {0: 'no_tumor', 1: 'not_brain', 2: 'tumor'}
index_to_class = {v: k for k, v in class_indices.items()}


@app.get("/")
def home():
    return {
        "message": "Medical Prediction API is running successfully with all models",
    }


class HealthInput(BaseModel):
    features: list


def _prepare_health_features(features):
    if not isinstance(features, list) or len(features) != 5:
        raise ValueError(
            "Expected a list of 5 features: [age_years, gender, heart_rate_bpm, spo2_percent, body_temperature_c]")

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
async def predict_brain(request: Request):
    content_type = request.headers.get("content-type", "")

    try:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            uploaded_file = form.get("file") or form.get("image")
            if uploaded_file is None:
                return JSONResponse(status_code=400,
                                    content={"error": "Please upload an image using the 'file' or 'image' form field."})
            image_bytes = await uploaded_file.read()
        else:
            image_bytes = await request.body()
            if not image_bytes:
                return JSONResponse(status_code=400, content={"error": "No image data received."})

        # Process image matching training preprocessing (224x224, rescale 1./255)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict using Keras model
        predictions = brain_model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        predicted_label = index_to_class.get(predicted_index, "Unknown")

        return {
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {index_to_class[i]: float(predictions[0][i]) for i in index_to_class}
        }
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Invalid image upload: {exc}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)