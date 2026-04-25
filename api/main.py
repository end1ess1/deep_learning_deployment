import io
import numpy as np
import pickle
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

import tensorflow as tf
from tensorflow import keras

app = FastAPI(title="Street Objects Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("best_model_11_efficientnetb0.keras")
ENCODER_PATH = Path("label_encoder.pkl")
IMAGE_SIZE = (224, 224)

model = None
label_encoder = None


@app.on_event("startup")
async def load_model():
    global model, label_encoder
    model = keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype="float32")  # EfficientNetB0 expects [0, 255]
    return np.expand_dims(arr, axis=0)


@app.get("/")
def root():
    return {"status": "ok", "model": "EfficientNetB0", "classes": list(label_encoder.classes_)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    img_array = preprocess(image_bytes)

    probs = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    predicted_class = label_encoder.classes_[class_idx]
    confidence = float(probs[class_idx])

    probabilities = {
        label_encoder.classes_[i]: float(probs[i])
        for i in range(len(label_encoder.classes_))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
