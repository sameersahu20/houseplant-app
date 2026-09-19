import os
import gc
import pathlib
import requests
import torch
import torchvision.transforms as T
from fastai.vision.all import *
import gradio as gr
from PIL import Image

# Fix: Allow Windows to load models trained on Linux/Kaggle
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl'

# Download model if not present locally
if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 10 * 1024 * 1024:
    print("Downloading model weights...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
    r.raise_for_status()
    with open(MODEL_FILE, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Load learner
learn = load_learner(MODEL_FILE)
learn.model.eval()

vocab = list(learn.dls.vocab)

transform_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    if img is None:
        return {}
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")

    with torch.no_grad():
        tensor = transform_pipeline(img).unsqueeze(0)
        logits = learn.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    del tensor, logits
    gc.collect()

    return {vocab[i]: float(probs[i]) for i in range(len(vocab))}

image_input = gr.Image(type="pil")

demo = gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=gr.Label(num_top_classes=3),
    title="Houseplant Classifier"
)

if __name__ == "__main__":
    demo.launch(share=True)