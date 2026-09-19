import os
import requests
import torch
from fastai.vision.all import *
import gradio as gr
from PIL import Image  # after fastai's import * so it isn't shadowed

torch.set_num_threads(1)  # avoid CPU thread contention on a small instance

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl'

if not os.path.exists(MODEL_FILE):
    print("Downloading model weights...")
    r = requests.get(MODEL_URL)
    with open(MODEL_FILE, 'wb') as f:
        f.write(r.content)
    print("Download complete.")

learn = load_learner(MODEL_FILE)
learn.model.eval()
vocab = list(learn.dls.vocab)

# Warm-up so the first real request isn't the slow one
learn.predict(Image.new("RGB", (224, 224)))

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