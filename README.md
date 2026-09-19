## 🌿🌱🪴 Houseplant Classifier 🪴🌱🌿

An end-to-end computer vision web application that identifies indoor houseplant varieties from uploaded photos in real time.

## 🌐 Live Application

The classifier is live and accessible globally:

[![Streamlit: houseplant-app](https://img.shields.io/badge/Streamlit-houseplant--app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://houseplant-app-pzpe4hju72mffdn2b5tqdt.streamlit.app/)

---

## 👨‍💻 Author

Sameer Sahu<br>
[![LinkedIn: Sameer Sahu](https://img.shields.io/badge/LinkedIn-Sameer%20Sahu-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sameersahu18/)<br>
[![GitHub: Sameer Sahu](https://img.shields.io/badge/GitHub-Sameer%20Sahu-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sameersahu20)

---

## 📖 The Story Behind the Project (Non-Technical Overview)

### What Does This App Do?

Imagine walking into a room or nursery, spotting an indoor plant you like, but having no idea what it is or how to care for it. This web app acts like a digital botanical field guide. You upload a picture of a houseplant from your phone or laptop, and within seconds, it tells you what species it is along with the model's confidence level.

### Where Did the Data Come From?

Computers cannot understand plants without seeing hundreds of examples first. The model was trained on a curated dataset of thousands of labeled photographs representing popular indoor plant species—photographed across various lighting conditions, angles, and stages of growth.

### How Does the Computer "Learn"?

Just like a person learns to recognize a Snake Plant by noticing tall, stiff, sword-like leaves with yellow edges, the artificial neural network examines thousands of leaf shapes, vein patterns, and surface textures. Over time, it adjusts internal mathematical weights until it can reliably tell distinct species apart—even on photos it has never seen before.

### How Is It Always Online?

The application is hosted 24/7 on cloud servers. Instead of running on a personal computer at home, the program lives in an automated cloud environment that accepts photo uploads from anywhere in the world, processes them through the trained model, and displays the result instantly.

---

## 📊 Dataset Details

- **Source:** [![Kaggle: House Plant Species Dataset](https://img.shields.io/badge/Kaggle-House%20Plant%20Species%20Dataset-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species/code)
- **Classes:** Image dataset containing **47 species** of indoor plants.
- **Dataset composition:** High-quality photographs of indoor houseplants categorized across diverse indoor lighting environments, pots, and growth stages.
- **Image preprocessing:** Images are standardized to 224×224 pixels with ImageNet normalization. FastAI transforms apply augmentations such as random flips, perspective shifts, and crops to improve inference generalization.

---

## 🛠️ Technical Architecture & Engineering Details

### Tech Stack

- **Framework & Deep Learning:** FastAI, PyTorch (`torch`, `torchvision`)
- **Web App & Interface:** Streamlit
- **Weight Hosting & Retrieval:** Hugging Face Hub
- **Deployment Platform:** Streamlit Community Cloud (Linux)
- **Image Preprocessing:** PIL (Python Imaging Library)

### Key Engineering Features

1. **Transfer Learning Pipeline:** Fine-tuned a deep convolutional vision backbone using FastAI to classify closed-set indoor plant varieties.
2. **Decoupled Model Weights:** To bypass GitHub repository file-size limitations, model weights (`.pkl`) are hosted externally on Hugging Face and dynamically retrieved via cached requests upon server spin-up.
3. **Cross-Platform Deserialization:** Implemented a runtime `pathlib` remapping bridge (`pathlib.WindowsPath = pathlib.PosixPath` and `sys.modules` patching) to resolve operating system serialization differences between Windows training environments and Linux cloud containers.
4. **Fluid, Responsive UI:** Custom CSS layer utilizing `clamp()` fluid typography and CSS safe-area margins (`env(safe-area-inset-bottom)`) to guarantee a uniform pill badge and controls across mobile devices and desktop viewports.
5. **Dynamic Vocabulary Inspection:** Interactive UI component inspecting the learner's vocabulary array dynamically to display supported taxonomy without hardcoded lists.

---

## 💻 Running Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sameersahu20/houseplant-app.git
   cd houseplant-app
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run streamlit_app.py
   ```

   The model is downloaded from Hugging Face automatically the first time the app starts.
