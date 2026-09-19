import os
import requests
from fastai.vision.all import *

MODEL_FILE = 'houseplant_model.pkl'
# Ensure this uses /resolve/main/ and not /blob/main/
MODEL_URL = 'https://huggingface.co/sameersahu20/houseplant_model/resolve/main/houseplant_model.pkl'

def download_model():
    # If file exists but is suspiciously small (under 10MB), delete and redownload
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) < 10 * 1024 * 1024:
        print("Detected incomplete/corrupted file. Removing...")
        os.remove(MODEL_FILE)

    if not os.path.exists(MODEL_FILE):
        print("Downloading model weights...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Download complete! File size: {os.path.getsize(MODEL_FILE) / (1024*1024):.2f} MB")

download_model()
learn = load_learner(MODEL_FILE)