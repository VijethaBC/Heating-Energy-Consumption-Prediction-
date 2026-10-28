from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
from typing import List, Dict, Optional
import os
from datetime import datetime

app = FastAPI(
    title="Building Energy Prediction API",
    description="API for predicting building energy consumption using various models",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    features: List[float]
    model_type: str = "random_forest"  # Options: random_forest, svr, lstm, cnn_lstm
    sequence_length: Optional[int] = None  # Required for LSTM and CNN-LSTM models

class PredictionOutput(BaseModel):
    prediction: float
    model_type: str
    timestamp: str

# Load models
def load_models():
    models = {}
    models_dir = "models"
    
    # Find the most recent model directory
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        raise RuntimeError("No model directories found")
    
    latest_dir = max(subdirs)
    model_dir = os.path.join(models_dir, latest_dir)
    
    # Load scikit-learn models
    for model_name in ["random_forest", "svr"]:
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Load tensorflow models
    for model_name in ["lstm_model", "cnn_lstm_model"]:
        keras_path = os.path.join(model_dir, f"{model_name}.keras")
        dir_path = os.path.join(models_dir, model_name)
        if os.path.exists(keras_path):
            models[model_name.replace("_model", "")] = tf.keras.models.load_model(keras_path)
        elif os.path.exists(dir_path):
            models[model_name.replace("_model", "")] = tf.keras.models.load_model(dir_path)
    
    return models

models = load_models()

@app.get("/")
async def root():
    return {"message": "Building Energy Prediction API"}

@app.get("/models")
async def get_available_models():
    return {"available_models": list(models.keys())}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if input_data.model_type not in models:
        raise HTTPException(status_code=400, detail=f"Model {input_data.model_type} not found")
    
    try:
        features = np.array(input_data.features)
        
        if input_data.model_type in ["lstm", "cnn_lstm"]:
            if input_data.sequence_length is None:
                raise HTTPException(status_code=400, detail="sequence_length is required for LSTM and CNN-LSTM models")
            
            # Reshape for sequential models
            features = features.reshape(1, input_data.sequence_length, -1)
            raw_pred = models[input_data.model_type].predict(features)
            prediction = float(np.squeeze(raw_pred))
        else:
            # Reshape for traditional models
            features = features.reshape(1, -1)
            raw_pred = models[input_data.model_type].predict(features)
            prediction = float(np.squeeze(raw_pred))
        
        return PredictionOutput(
            prediction=prediction,
            model_type=input_data.model_type,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 