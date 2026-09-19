import os
import requests
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import *

st.set_page_config(page_title="Houseplant Classifier", page_icon="🌿")

st.title("🌿 Houseplant Classifier")
st.write("Upload a photo of a houseplant to identify its species.")

MODEL_FILE = 'houseplant_model.pkl'
MODEL_URL = 'https://huggingface.co/sameersahu21/houseplant-classifier-model/resolve/main/houseplant_model.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 10 * 1024 * 1024:
        with st.spinner("Downloading model weights..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
            r.raise_for_status()
            with open(MODEL_FILE, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    
    learn = load_learner(MODEL_FILE)
    learn.model.eval()
    return learn, list(learn.dls.vocab)

learn, vocab = load_model()

transform_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose a houseplant image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with torch.no_grad():
        tensor = transform_pipeline(image).unsqueeze(0)
        logits = learn.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk_probs, topk_indices = torch.topk(probs, 3)

    st.subheader("Predictions")
    for prob, idx in zip(topk_probs, topk_indices):
        st.write(f"**{vocab[idx]}**: {prob.item() * 100:.1f}%")
        st.progress(float(prob.item()))