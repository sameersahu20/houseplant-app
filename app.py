import os
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
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

image = gr.Image(type="pil")
label = gr.Label(num_top_classes=3)

demo = gr.Interface(
    fn=predict,
    inputs=image,
    outputs=label,
    title="Houseplant Identifier",
    description="Take a photo of a leaf or upload an image to identify the houseplant species."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))