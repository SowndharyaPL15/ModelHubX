import os
import pickle
from fastapi import FastAPI, HTTPException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_server")

app = FastAPI(title="ModelHubX Inference Pod")

MODEL_PATH = os.getenv("MODEL_PATH", "/models/default.pkl")
model_instance = None
model_metadata = {}

@app.on_event("startup")
def load_model():
    global model_instance, model_metadata
    logger.info(f"Loading generic inference model from: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                bundled = pickle.load(f)
                model_instance = bundled.get("model")
                model_metadata = bundled.get("metadata", {})
            logger.info(f"Successfully loaded '{model_metadata.get('name')}' v{model_metadata.get('version')}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"MODEL_PATH not found: {MODEL_PATH}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_instance is not None}

@app.get("/metadata")
def metadata():
    return model_metadata

@app.post("/predict")
def predict(input_data: list):
    """
    Standard generic inference router mapping JSON arrays to predict() calls.
    Equivalent to Sagemaker / Triton endpoints.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")
    
    try:
        predictions = model_instance.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
