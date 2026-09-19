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
st.title("🌿🌱🪴 Houseplant Classifier 🪴🌱🌿")
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
with st.expander(f"📋 View all {len(vocab)} supported plant species"):
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
footer_html = """
<style>
.custom-footer {
    position: fixed;
    left: 50%;
    bottom: max(16px, env(safe-area-inset-bottom, 16px));
    transform: translateX(-50%);
    background-color: rgba(20, 20, 24, 0.92);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 9999px;
    padding: clamp(6px, 1.5vw, 10px) clamp(14px, 3vw, 24px);
    display: flex !important;
    flex-direction: row !important;
    align-items: center;
    justify-content: center;
    gap: clamp(8px, 2vw, 14px);
    font-size: clamp(12px, 2.5vw, 15px);
    font-weight: 500;
    color: #f1f1f1;
    z-index: 99999;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
    white-space: nowrap !important;
    width: max-content !important;
    max-width: 95vw;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    user-select: none;
    -webkit-user-select: none;
}
.custom-footer span {
    white-space: nowrap !important;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.custom-footer a {
    display: inline-flex;
    align-items: center;
    color: #d1d5db;
    transition: color 0.2s ease, transform 0.2s ease;
    text-decoration: none;
    flex-shrink: 0;
}
.custom-footer a:hover {
    color: #38bdf8;
    transform: scale(1.15);
}
.custom-footer svg {
    width: clamp(16px, 2.8vw, 20px);
    height: clamp(16px, 2.8vw, 20px);
    fill: currentColor;
    flex-shrink: 0;
}
</style>

<div class="custom-footer">
    <span>Made with ❤️ by <strong>Sameer Sahu</strong></span>
    <!-- LinkedIn -->
    <a href="https://www.linkedin.com/in/sameersahu18/" target="_blank" rel="noopener noreferrer" title="LinkedIn Profile">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.46 10.9v8.37H9.2V10.9H6.46M7.83 6.22a1.62 1.62 0 1 0 0 3.24 1.62 1.62 0 0 0 0-3.24Z"/>
        </svg>
    </a>
    <!-- GitHub -->
    <a href="https://github.com/sameersahu20" target="_blank" rel="noopener noreferrer" title="GitHub Profile">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2Z"/>
        </svg>
    </a>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)