import os
import sys
import pathlib
import platform

# Cross-platform compatibility patch for models saved on Windows
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
    # Python 3.13/3.14 unpickler lookup compatibility
    sys.modules['pathlib.WindowsPath'] = pathlib.PosixPath

import requests
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import *

st.set_page_config(page_title="Houseplant Classifier", page_icon="🌿")
st.title("🪴 Houseplant Classifier 🪴")
st.write("Upload a photo of a houseplant for me to identify its species.")


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

# --- Supported Species List ---
with st.expander(f"🌿📋 The {len(learn.dls.vocab)} plant species that I can identify!"):
    sorted_species = sorted(vocab)
    # Split the names evenly across 2 columns
    col1, col2 = st.columns(2)
    mid = (len(sorted_species) + 1) // 2
    
    with col1:
        for plant in sorted_species[:mid]:
            st.markdown(f"• {plant}")
    with col2:
        for plant in sorted_species[mid:]:
            st.markdown(f"• {plant}")

st.caption("Want to read about the dataset & architecture? [View project on GitHub](https://github.com/sameersahu20/houseplant-app)")

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


# --- Universal Responsive Footer ---
st.markdown(
    """
    <style>
    /* Fixed bottom-left badge */
    .footer-badge {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 999;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(22, 27, 34, 0.88);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
        font-size: 13px;
        color: #e6edf3;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .footer-badge a {
        color: #58a6ff;
        display: inline-flex;
        align-items: center;
        text-decoration: none;
        margin-left: 4px;
        transition: color 0.15s ease;
    }

    .footer-badge a:hover {
        color: #79c0ff;
    }

    /* Mobile fine-tuning */
    @media (max-width: 640px) {
        .footer-badge {
            left: 12px;
            bottom: calc(16px + env(safe-area-inset-bottom));
            padding: 6px 12px;
            font-size: 12px;
            max-width: calc(100vw - 120px);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)