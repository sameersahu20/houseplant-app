import os
import gc
import requests
import torch
from fastai.vision.all import *
import gradio as gr
from PIL import Image

# 1. Hugging Face Model Download Setup
MODEL_FILE = 'houseplant_model.pkl'
# Replace this URL with your exact Hugging Face raw model link if different
MODEL_URL = 'https://huggingface.co/sameersahu20/houseplant_model/resolve/main/houseplant_model.pkl'

if not os.path.exists(MODEL_FILE):
    print("Downloading model weights...")
    r = requests.get(MODEL_URL)
    with open(MODEL_FILE, 'wb') as f:
        f.write(r.content)
    print("Download complete.")

# 2. Load the exported FastAI learner
learn = load_learner(MODEL_FILE)

# 3. Memory-Optimized Prediction Function
def predict(img):
    # Ensure input is a PIL image and force-resize to 224x224
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB").resize((224, 224))

    # Disable autograd calculations to stay under the 512 MB RAM ceiling
    with torch.no_grad():
        pred, pred_idx, probs = learn.predict(img)

    # Force Python garbage collection after inference
    gc.collect()

    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# 4. Gradio Interface
image_input = gr.Image(type="pil")

demo = gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=gr.Label(num_top_classes=3),
    title="Houseplant Classifier"
)

# 5. Bind dynamically to Render's assigned port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)