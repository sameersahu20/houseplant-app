import os
import requests
import torch
from fastai.vision.all import *
import gradio as gr
from PIL import Image  # after fastai's import * so it isn't shadowed

torch.set_num_threads(1)  # avoid CPU thread contention on a small instance

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu20/houseplant_model/resolve/main/houseplant_model.pkl'

if not os.path.exists(MODEL_FILE):
    print("Downloading model weights...")
    with requests.get(MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()  # fail loudly on 401/404 instead of saving an error page
        with open(MODEL_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    print("Download complete.")

learn = load_learner(MODEL_FILE, cpu=True)
learn.model.eval()
vocab = list(learn.dls.vocab)

# Warm-up so the first real request isn't the slow one
learn.predict(Image.new("RGB", (224, 224)))

def predict(img):
    if img is None:
        return {}
    img = img.convert("RGB")
    img.thumbnail((512, 512))  # shrink huge phone photos, keep aspect ratio
    _, _, probs = learn.predict(img)
    return {vocab[i]: float(probs[i]) for i in range(len(vocab))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Houseplant Classifier",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.queue(max_size=5).launch(server_name="0.0.0.0", server_port=port)