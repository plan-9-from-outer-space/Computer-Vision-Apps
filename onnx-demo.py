
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
# from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
from contextlib import asynccontextmanager

ONNX_MODEL_PATH = "Models/vit_classification.onnx" # Downloaded from Colab

ort_session = None # Global variable to hold the model session

# Define lifespan event to load the model at startup
@asynccontextmanager
async def lifespan (app: FastAPI):
    global ort_session
    # Only load the model once when the app starts
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    yield
    ort_session = None

# Create FastAPI app with lifespan
app = FastAPI (lifespan=lifespan)

# Mount the "static" directory to the "/static" path in the app
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Image transforms
def preprocess_image (image_bytes):

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_img = img.resize((224, 224))
    np_array = np.asarray(resized_img)
    img_transposed = np.transpose(np_array, (2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)
    img_normalized = img_expanded.astype(np.float32) / 255.0
    return img_normalized

# Inference endpoint
@app.post("/inference/")
async def inference (file: UploadFile = File(...)):

    if ort_session is None:
        return {"error": "Model is not loaded"}
    image_bytes = await file.read()
    image_preprocessed = preprocess_image (image_bytes)

    session_input = {ort_session.get_inputs()[0].name: image_preprocessed}
    onnx_output = ort_session.run(None, session_input)
    onnx_logits = onnx_output[0]

    pred_idx = int(np.argmax(onnx_logits, axis=1)[0])

    return {"predicted_idx": pred_idx}

# Test endpoint
@app.get("/")
def test_api():
    return {"message": "the api is live !!!"}
