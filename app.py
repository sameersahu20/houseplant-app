import os
import gc
import requests
import torch
import torchvision.transforms as T
from fastai.vision.all import *
import gradio as gr
from PIL import Image

# Prevent PyTorch/OpenMP multi-threading segfaults on single-core containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl'

# 1. Download model weights if missing or incomplete (< 10MB)
if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 10 * 1024 * 1024:
    print("Downloading model weights...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
    r.raise_for_status()
    with open(MODEL_FILE, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Download complete! File size: {os.path.getsize(MODEL_FILE) / (1024*1024):.2f} MB")

# 2. Load model
learn = load_learner(MODEL_FILE)
learn.model.eval()

# Extract vocabulary labels directly
vocab = list(learn.dls.vocab)

# Standard ImageNet normalization used by FastAI vision models
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

    # Pure PyTorch inference bypasses FastAI DataLoader memory bloat and segfaults
    with torch.no_grad():
        tensor = transform_pipeline(img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        logits = learn.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Clean up memory immediately
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
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)