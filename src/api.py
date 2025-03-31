from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from torchvision import transforms
from models.model import MyModel
import uvicorn
import numpy as np
import os
from download_model import download_model

app = FastAPI(
    title="Bolt Hole Cast Number Identification API",
    description="API for identifying cast numbers in bolt holes on hub surfaces",
    version="1.0.0"
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(100)  # Initialize your model

# Download model file if it doesn't exist
if not os.path.exists('best_model.pth'):
    download_model()

# Load the trained weights
try:
    checkpoint = torch.load('best_model.pth', map_location=device)
    # Try different ways to load the model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.get("/")
def home():
    return {
        "message": "Bolt Hole Cast Number Identification API",
        "usage": "Send a POST request to /predict with an image file to identify the cast number"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read and transform image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Add size limit
        if image.size[0] * image.size[1] > 4096 * 4096:
            raise HTTPException(status_code=400, detail="Image too large")
            
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            prediction = predicted.item()
            confidence = confidence.item()
        
        # Convert prediction to result
        if prediction == 100:
            result = "No cast number detected"
        else:
            result = str(prediction).zfill(2)  # Format as two digits
        
        return {
            "prediction": result,
            "confidence": float(confidence),
            "raw_prediction": int(prediction),
            "image_size": image.size,
            "processing_device": str(device)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 