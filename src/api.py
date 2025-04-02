from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
from torchvision import transforms
from models.model import MyModel
import uvicorn

app = FastAPI(
    title="Bolt Hole Number Recognition API",
    description="API for recognizing numbers in bolt hole images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize model
model = MyModel(100).to(device)

# Load the trained model
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_image(image_bytes):
    """Preprocess the uploaded image."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        # Apply transformations
        image_tensor = transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Bolt Hole Number Recognition API",
        "version": "1.0.0",
        "status": "active",
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for predicting numbers in bolt hole images.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        image_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        # Prepare response
        response = {
            "predicted_number": predicted_class if predicted_class < 100 else "no_number",
            "confidence": float(confidence),
            "raw_probabilities": probabilities[0].tolist()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 