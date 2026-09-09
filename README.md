# 🎨 Stable Diffusion Local — Photorealistic AI Studio

A self-hosted, GPU-accelerated web studio for photorealistic image generation and image-to-image editing, powered by **Realistic Vision v5.1** (SD 1.5) and **RealVisXL v4.0** (SDXL).

Features specialized **Face & Identity Preservation Mode** designed for high-accuracy face and body consistency when restyling outfits, backgrounds, or artistic styles.

---

## ✨ Features

- 🚀 **Dual Model Engine**:
  - **Realistic Vision v5.1 (SD 1.5)**: High-speed, photorealistic portraits and scenes with low VRAM footprint.
  - **RealVisXL v4.0 (SDXL)**: Next-generation photorealism, ultra-fine skin textures, intricate clothing details, and 1024x1024 native resolution.
  - Seamless on-the-fly model switching with automated VRAM memory recycling.
- 👤 **Face & Identity Preservation Mode**:
  - Maintains subject identity, facial structure, skin tone, and body proportions during img2img modifications (e.g. changing clothes, hair styling, scenery change).
  - Dual control: Balance Transformation Strength vs Identity Retention with precision sliders.
- 🎯 **High Prompt Accuracy & Smart Negative Prompts**:
  - Automatically enriches prompts with realism anchors (lens details, studio lighting).
  - Built-in negative prompt suite to eliminate artifacts, warped hands, and over-saturation.
- 💻 **Modern Glassmorphic UI**:
  - Responsive, dark-themed dashboard.
  - Drag-and-drop reference image uploader.
  - Side-by-side original vs generated comparison.
  - One-click copy prompt and image download.
- ⚡ **Automated 1-Click Setup**:
  - Fully automated bash script handles virtual environment creation, CUDA PyTorch installation, model weights downloading, and service startup.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask, HuggingFace `diffusers`, `transformers`, `accelerate`
- **Inference Engine**: PyTorch with CUDA 12.1+ / cuDNN acceleration & CPU fallback
- **Frontend**: Vanilla HTML5, Modern CSS (Glassmorphism & Flexbox/Grid), Vanilla JavaScript (ES6+)

---

## 🚀 Quick Start (Automated Setup)

Clone and launch the entire studio in one command on Ubuntu/Debian:

```bash
git clone https://github.com/nadunMadhushanka/test-sdf.git
cd test-sdf
chmod +x setup_stable_diffusion.sh
./setup_stable_diffusion.sh
```

The script will automatically:
1. Detect your GPU (NVIDIA CUDA) or set up CPU fallback.
2. Create an isolated Python virtual environment (`venv`).
3. Install PyTorch, Diffusers, and required machine learning dependencies.
4. Download the pre-trained weights for Realistic Vision v5.1 and RealVisXL v4.0.
5. Create a desktop application shortcut.
6. Launch the server at **`http://localhost:7860`**.

---

## 🖥️ Manual Installation

If you prefer to set up manually:

### 1. Clone the repository
```bash
git clone https://github.com/nadunMadhushanka/test-sdf.git
cd test-sdf
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
# For NVIDIA GPUs with CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install diffusers transformers accelerate safetensors flask flask-cors pillow requests
```

### 4. Download Models
Place the models in the `models/` directory:
- `models/realisticVisionV51_v51VAE.safetensors`
- `models/realvisxlV40_v40Bakedvae.safetensors`

### 5. Launch the Web UI
```bash
python3 server.py
```
Open your browser and navigate to `http://localhost:7860`.

---

## 📖 Usage Guide

### Text-to-Image (txt2img)
1. Select your model: **Realistic Vision (SD 1.5)** for fast iterations or **RealVisXL (SDXL)** for max detail.
2. Enter your positive prompt describing the subject, environment, and lighting.
3. Choose resolution, sampling steps (25–35 recommended), and CFG scale (5.0–7.5).
4. Click **Generate Image**.

### Image-to-Image with Face Preservation (img2img)
1. Upload a portrait or full-body reference image.
2. Enable **Preserve Face & Body Shape** toggle.
3. Set **Transformation Strength** (e.g., `0.35` - `0.55` allows outfit/background changes while preserving the face).
4. Enter your prompt (e.g. `wearing a red summer dress, sunset beach background, 8k portrait`).
5. Click **Generate**.

---

## 📁 Project Structure

```
test-sdf/
├── models/                     # Model weights (.safetensors - excluded from git)
├── outputs/                    # Generated images directory
├── static/                     # Web UI frontend assets
│   ├── index.html              # Studio interface
│   ├── style.css               # Glassmorphism dark mode stylesheet
│   └── app.js                  # Frontend controller & API bridge
├── server.py                   # Flask backend & diffusers pipeline manager
├── setup_stable_diffusion.sh   # 1-click automated setup and installer
├── run_sd.sh                   # Launcher script for desktop shortcut
├── test_gen.py                 # Smoke test script for pipeline verification
├── .gitignore                  # Git rules to exclude caches and large binaries
└── README.md                   # Documentation
```

---

## 📜 License

This project is open-source under the MIT License. Models are subject to their respective licenses:
- [Realistic Vision v5.1 License](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE)
- [RealVisXL v4.0 License](https://huggingface.co/SG161222/RealVisXL_V4.0)
