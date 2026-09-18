import os
import torch
import urllib.request
import gradio as gr
from fastai.vision.all import *

MODEL_URL = "https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl"
MODEL_FILE = "houseplant_model.pkl"

# Download the model weights on server start if not present
if not os.path.exists(MODEL_FILE):
    print("Downloading model weights...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download complete.")

# Load model and class names
learn = load_learner(MODEL_FILE)
labels = learn.dls.vocab

def predict(img):
    if isinstance(img, Image.Image):
        img = img.resize((224, 224))
    else:
        img = Image.fromarray(img).resize((224, 224))

    pred, pred_idx, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

image_input = gr.Image(type="pil")

demo = gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=gr.Label(num_top_classes=3),
    title="Houseplant Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=10000)