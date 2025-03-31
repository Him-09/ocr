import os
import requests
from pathlib import Path
import gdown

def download_model():
    """Download the model file from Google Drive."""
    # Google Drive file ID (you'll need to upload your model to Google Drive and get this ID)
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    model_path = Path("best_model.pth")
    
    if not model_path.exists():
        print("Downloading model file...")
        try:
            # Download from Google Drive
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, str(model_path), quiet=False)
            print("Model file downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("Model file already exists!")

if __name__ == "__main__":
    download_model() 