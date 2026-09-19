import os
import gc
import requests
import torch
from fastai.vision.all import *
import gradio as gr
from PIL import Image

# Cap CPU threads so PyTorch doesn't spin up heavy thread pools
torch.set_num_threads(1)

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl'

# Download weights if not present or incomplete (< 10MB)
if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 10 * 1024 * 1024:
    print("Downloading model weights...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_FILE, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

learn = load_learner(MODEL_FILE)
learn.model.eval()

# No warm-up prediction here — preserve RAM for server startup!

def predict(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize((224, 224))

    # Disable gradient tracking to prevent RAM spikes
    with torch.no_grad():
        pred, pred_idx, probs = learn.predict(img)

    # Force Python to clean temporary memory immediately
    gc.collect()

    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

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