#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Stable Diffusion Local — Full Automated Setup
#  Sets up everything from scratch on a fresh machine with an NVIDIA GPU.
#
#  What this script does:
#    1. Clones/creates the project directory
#    2. Creates Python virtual environment
#    3. Installs PyTorch (CUDA 12.1) + diffusers + transformers + FastAPI
#    4. Downloads Realistic Vision v5.1 (SD 1.5, ~2 GB)
#    5. Downloads RealVisXL v4.0 (SDXL, ~6.5 GB)
#    6. Creates all SDXL HuggingFace cache config files (offline compatibility)
#    7. Creates the web server (server.py), frontend (HTML/CSS/JS)
#    8. Creates desktop shortcut + run script
#    9. Starts the server
#
#  Usage:
#    chmod +x setup_stable_diffusion.sh
#    ./setup_stable_diffusion.sh
#
#  Requirements:
#    - Ubuntu/Debian Linux
#    - NVIDIA GPU with drivers installed (nvidia-smi must work)
#    - Python 3.10+
#    - Internet connection (for initial setup only)
#    - ~12 GB disk space (models + packages)
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="${HOME}/Desktop/stable-diffusion"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/models"
STATIC_DIR="${PROJECT_DIR}/static"
OUTPUTS_DIR="${PROJECT_DIR}/outputs"
SERVER_PORT=7860

# Model URLs (HuggingFace)
SD15_URL="https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
SD15_FILE="Realistic_Vision_V5.1.safetensors"
SDXL_URL="https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors"
SDXL_FILE="RealVisXL_V4.0.safetensors"

# SDXL HuggingFace cache path (diffusers expects configs here)
SDXL_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ── Helper functions ──────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[  OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0: Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════════
step "Step 0/9: Pre-flight checks"

# Check NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
info "GPU detected: ${GPU_NAME} (${GPU_VRAM})"

# Check Python
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.10+ first."
fi
PYTHON_VER=$(python3 --version 2>&1)
info "Python: ${PYTHON_VER}"

# Check pip
if ! python3 -m pip --version &>/dev/null; then
    warn "pip not found, installing..."
    sudo apt-get update && sudo apt-get install -y python3-pip python3-venv
fi

# Check git
if ! command -v git &>/dev/null; then
    warn "git not found, installing..."
    sudo apt-get update && sudo apt-get install -y git
fi

# Check curl
if ! command -v curl &>/dev/null; then
    warn "curl not found, installing..."
    sudo apt-get update && sudo apt-get install -y curl
fi

# Check disk space (need ~12 GB)
AVAIL_GB=$(df -BG "${HOME}" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "${AVAIL_GB}" -lt 12 ]; then
    warn "Only ${AVAIL_GB} GB available. Need ~12 GB. Proceeding anyway..."
else
    info "Disk space: ${AVAIL_GB} GB available ✓"
fi

success "Pre-flight checks passed"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Create project directory
# ══════════════════════════════════════════════════════════════════════════════
step "Step 1/9: Creating project directory"

mkdir -p "${PROJECT_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${STATIC_DIR}"
mkdir -p "${OUTPUTS_DIR}"

cd "${PROJECT_DIR}"
success "Project directory: ${PROJECT_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Create Python virtual environment
# ══════════════════════════════════════════════════════════════════════════════
step "Step 2/9: Setting up Python virtual environment"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    info "Created new venv at ${VENV_DIR}"
else
    info "Existing venv found, reusing"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q
success "Virtual environment ready"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Install Python packages
# ══════════════════════════════════════════════════════════════════════════════
step "Step 3/9: Installing Python packages (PyTorch + diffusers + FastAPI)"

info "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

info "Installing diffusers, transformers, accelerate..."
pip install diffusers transformers accelerate -q

info "Installing FastAPI + Uvicorn + Pillow..."
pip install fastapi uvicorn pillow numpy pydantic python-multipart -q

# Verify torch + CUDA
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || error "PyTorch installation failed"

success "All packages installed"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Download AI models
# ══════════════════════════════════════════════════════════════════════════════
step "Step 4/9: Downloading AI models"

download_model() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "${dest}" ]; then
        local size
        size=$(stat -c%s "${dest}" 2>/dev/null || echo 0)
        if [ "${size}" -gt 1000000000 ]; then
            info "${name} already downloaded ($(echo "scale=2; ${size}/1073741824" | bc) GB), skipping"
            return 0
        fi
    fi

    info "Downloading ${name}... (this may take 5-15 minutes)"
    curl -L -C - --retry 5 --progress-bar \
        -o "${dest}" "${url}" || {
        warn "Download failed for ${name}, retrying..."
        curl -L --retry 10 --progress-bar -o "${dest}" "${url}"
    }
    success "${name} downloaded"
}

# SD 1.5 — Realistic Vision v5.1 (~2 GB)
download_model "${SD15_URL}" "${MODELS_DIR}/${SD15_FILE}" "Realistic Vision v5.1 (SD 1.5)"

# SDXL — RealVisXL v4.0 (~6.5 GB)
download_model "${SDXL_URL}" "${MODELS_DIR}/${SDXL_FILE}" "RealVisXL v4.0 (SDXL)"

success "All models downloaded"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Create SDXL HuggingFace cache configs (offline compatibility)
# ══════════════════════════════════════════════════════════════════════════════
step "Step 5/9: Creating SDXL offline config files"

mkdir -p "${SDXL_CACHE_DIR}/scheduler"
mkdir -p "${SDXL_CACHE_DIR}/unet"
mkdir -p "${SDXL_CACHE_DIR}/vae"
mkdir -p "${SDXL_CACHE_DIR}/text_encoder"
mkdir -p "${SDXL_CACHE_DIR}/text_encoder_2"

# model_index.json
cat > "${SDXL_CACHE_DIR}/model_index.json" << 'JSONEOF'
{
  "_class_name": "StableDiffusionXLPipeline",
  "_diffusers_version": "0.25.0",
  "force_zeros_for_empty_prompt": true,
  "add_watermarker": false,
  "scheduler": ["diffusers", "EulerDiscreteScheduler"],
  "text_encoder": ["transformers", "CLIPTextModel"],
  "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
  "tokenizer": ["transformers", "CLIPTokenizer"],
  "tokenizer_2": ["transformers", "CLIPTokenizer"],
  "unet": ["diffusers", "UNet2DConditionModel"],
  "vae": ["diffusers", "AutoencoderKL"]
}
JSONEOF

# scheduler_config.json
cat > "${SDXL_CACHE_DIR}/scheduler/scheduler_config.json" << 'JSONEOF'
{
  "_class_name": "EulerDiscreteScheduler",
  "_diffusers_version": "0.25.0",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "interpolation_type": "linear",
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "sample_max_value": 1.0,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "use_karras_sigmas": false
}
JSONEOF

# unet/config.json
cat > "${SDXL_CACHE_DIR}/unet/config.json" << 'JSONEOF'
{
  "_class_name": "UNet2DConditionModel",
  "_diffusers_version": "0.25.0",
  "act_fn": "silu",
  "addition_embed_type": "text_time",
  "addition_embed_type_num_heads": 64,
  "addition_time_embed_dim": 256,
  "attention_head_dim": [5, 10, 20],
  "block_out_channels": [320, 640, 1280],
  "center_input_sample": false,
  "cross_attention_dim": 2048,
  "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
  "downsample_padding": 1,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_type": "UNetMidBlock2DCrossAttn",
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "out_channels": 4,
  "projection_class_embeddings_input_dim": 2816,
  "sample_size": 128,
  "transformer_layers_per_block": [1, 2, 10],
  "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
  "use_linear_projection": true
}
JSONEOF

# vae/config.json
cat > "${SDXL_CACHE_DIR}/vae/config.json" << 'JSONEOF'
{
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.25.0",
  "act_fn": "silu",
  "block_out_channels": [128, 256, 512, 512],
  "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
  "force_upcast": true,
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.13025,
  "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"]
}
JSONEOF

# text_encoder/config.json
cat > "${SDXL_CACHE_DIR}/text_encoder/config.json" << 'JSONEOF'
{
  "_class_name": "CLIPTextModel",
  "architectures": ["CLIPTextModel"],
  "hidden_act": "quick_gelu",
  "hidden_size": 768,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 77,
  "model_type": "clip_text_model",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "projection_dim": 768,
  "vocab_size": 49408
}
JSONEOF

# text_encoder_2/config.json
cat > "${SDXL_CACHE_DIR}/text_encoder_2/config.json" << 'JSONEOF'
{
  "_class_name": "CLIPTextModelWithProjection",
  "architectures": ["CLIPTextModelWithProjection"],
  "hidden_act": "gelu",
  "hidden_size": 1280,
  "intermediate_size": 5120,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 77,
  "model_type": "clip_text_model",
  "num_attention_heads": 20,
  "num_hidden_layers": 32,
  "projection_dim": 1280,
  "vocab_size": 49408
}
JSONEOF

success "SDXL offline configs created"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Create application source files
# ══════════════════════════════════════════════════════════════════════════════
step "Step 6/9: Creating application source files"

# ── server.py ─────────────────────────────────────────────────────────────────
cat > "${PROJECT_DIR}/server.py" << 'PYEOF'
import os
import io
import time
import base64
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Strict offline environment enforcement
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

app = FastAPI(title="Stable Diffusion Local (SD 1.5 + SDXL)")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SD15_MODEL_PATH = os.path.join(MODELS_DIR, "Realistic_Vision_V5.1.safetensors")
SDXL_MODEL_PATH = os.path.join(MODELS_DIR, "RealVisXL_V4.0.safetensors")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Pipelines registry
pipelines = {
    "sd15": {"txt2img": None, "img2img": None},
    "sdxl": {"txt2img": None, "img2img": None}
}
load_errors = {}

def get_latent_rgb_factors():
    return torch.tensor([
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473]
    ], dtype=torch.float32)

def decode_latents_without_vae(latents: torch.Tensor) -> Image.Image:
    """Decodes SD latents into an image without using the VAE neural network decoder."""
    latents = latents.detach().cpu().to(torch.float32)
    latent_tensor = latents[0].permute(1, 2, 0)
    rgb_factors = get_latent_rgb_factors()
    rgb = torch.matmul(latent_tensor, rgb_factors)
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max - rgb_min > 1e-5:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb_np = (rgb.numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(rgb_np)
    image = image.resize((image.width * 8, image.height * 8), Image.Resampling.LANCZOS)
    return image

def load_sd15():
    if pipelines["sd15"]["txt2img"] is not None:
        return pipelines["sd15"]
    if not os.path.exists(SD15_MODEL_PATH):
        raise FileNotFoundError(f"SD 1.5 model not found at {SD15_MODEL_PATH}")
    print(f"Loading Realistic Vision v5.1 from {SD15_MODEL_PATH}...")
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_single_file(
        SD15_MODEL_PATH,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
    img2img.scheduler = pipe.scheduler
    img2img.to(device)
    if device == "cuda":
        img2img.enable_attention_slicing()
    pipelines["sd15"]["txt2img"] = pipe
    pipelines["sd15"]["img2img"] = img2img
    print("Realistic Vision v5.1 loaded successfully!")
    return pipelines["sd15"]

def load_sdxl():
    if pipelines["sdxl"]["txt2img"] is not None:
        return pipelines["sdxl"]
    if not os.path.exists(SDXL_MODEL_PATH):
        raise FileNotFoundError(f"SDXL model not found at {SDXL_MODEL_PATH}")
    print(f"Loading RealVisXL v4.0 from {SDXL_MODEL_PATH}...")
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_single_file(
        SDXL_MODEL_PATH,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    )
    pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    img2img.scheduler = pipe.scheduler
    img2img.to(device)
    if device == "cuda":
        img2img.enable_attention_slicing()
    pipelines["sdxl"]["txt2img"] = pipe
    pipelines["sdxl"]["img2img"] = img2img
    print("RealVisXL v4.0 loaded successfully!")
    return pipelines["sdxl"]


# ─── Default prompts ──────────────────────────────────────────────────────────
FACE_NEGATIVE = (
    "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, "
    "drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, "
    "bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, "
    "mutation, ugly, disgusting, amputation, bad eyes, cross-eyed, squinting, asymmetrical eyes, "
    "bad teeth, open mouth without expression, grainy, duplicate, watermark, signature, text, blurry, "
    "out of focus, overexposed, underexposed, multiple faces, extra head, cloned face"
)


class GenerateRequest(BaseModel):
    model_type: Optional[str] = "sd15"
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = 30
    cfg_scale: Optional[float] = 7.5
    seed: Optional[int] = -1
    width: Optional[int] = 512
    height: Optional[int] = 512
    disable_vae: Optional[bool] = False
    image_base64: Optional[str] = None
    strength: Optional[float] = 0.50
    face_preserve: Optional[bool] = False


@app.on_event("startup")
def startup_event():
    if os.path.exists(SD15_MODEL_PATH):
        try:
            load_sd15()
        except Exception as e:
            print(f"SD15 startup load deferred: {e}")


@app.get("/api/models")
def get_models():
    sdxl_size = None
    sdxl_complete = False
    if os.path.exists(SDXL_MODEL_PATH):
        size_bytes = os.path.getsize(SDXL_MODEL_PATH)
        sdxl_size = f"{size_bytes / (1024**3):.2f} GB"
        sdxl_complete = size_bytes > 6_400_000_000
    return {
        "models": [
            {
                "id": "sd15",
                "name": "Realistic Vision v5.1 (SD 1.5)",
                "resolution": "512x512",
                "available": os.path.exists(SD15_MODEL_PATH),
                "loaded": pipelines["sd15"]["txt2img"] is not None,
                "description": "Fast generation (~3s), portrait-optimized"
            },
            {
                "id": "sdxl",
                "name": "RealVisXL v4.0 (SDXL)",
                "resolution": "1024x1024",
                "available": os.path.exists(SDXL_MODEL_PATH),
                "loaded": pipelines["sdxl"]["txt2img"] is not None,
                "description": "Native 1024p, dual OpenCLIP, highest prompt accuracy",
                "download_size": sdxl_size,
                "download_complete": sdxl_complete
            }
        ]
    }


@app.get("/api/status")
def get_status():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_free = None
    vram_total = None
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        vram_free = f"{free_bytes / (1024**3):.1f} GB"
        vram_total = f"{total_bytes / (1024**3):.1f} GB"
    sdxl_size = 0
    sdxl_complete = False
    if os.path.exists(SDXL_MODEL_PATH):
        sdxl_size = os.path.getsize(SDXL_MODEL_PATH)
        sdxl_complete = sdxl_size > 6_400_000_000
    return {
        "device": device_name,
        "vram_free": vram_free,
        "vram_total": vram_total,
        "sd15_ready": os.path.exists(SD15_MODEL_PATH),
        "sd15_loaded": pipelines["sd15"]["txt2img"] is not None,
        "sdxl_ready": os.path.exists(SDXL_MODEL_PATH),
        "sdxl_loaded": pipelines["sdxl"]["txt2img"] is not None,
        "sdxl_download_complete": sdxl_complete,
        "sdxl_download_gb": round(sdxl_size / (1024**3), 2),
        "offline_mode": True
    }


@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    model_choice = req.model_type or "sd15"

    if model_choice == "sdxl":
        if not os.path.exists(SDXL_MODEL_PATH):
            raise HTTPException(status_code=400, detail="SDXL model not found.")
        size_bytes = os.path.getsize(SDXL_MODEL_PATH)
        if size_bytes < 6_400_000_000:
            raise HTTPException(
                status_code=400,
                detail=f"SDXL model is still downloading ({size_bytes/(1024**3):.2f} GB / ~6.6 GB). Please wait."
            )
        pipe_bundle = load_sdxl()
    else:
        model_choice = "sd15"
        pipe_bundle = load_sd15()

    txt2img_pipe = pipe_bundle["txt2img"]
    img2img_pipe = pipe_bundle["img2img"]

    if req.seed is None or req.seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    else:
        seed = int(req.seed)

    generator = torch.Generator(device=txt2img_pipe.device).manual_seed(seed)
    start_time = time.time()

    is_img2img = bool(req.image_base64 and len(req.image_base64.strip()) > 10)
    print(f"Generate [{model_choice}]: mode={'img2img' if is_img2img else 'txt2img'}, "
          f"face_preserve={req.face_preserve}, strength={req.strength}, "
          f"cfg={req.cfg_scale}, steps={req.steps}, seed={seed}, "
          f"prompt='{req.prompt[:50]}...'")

    final_prompt = req.prompt
    final_negative = req.negative_prompt if req.negative_prompt else FACE_NEGATIVE

    if req.face_preserve or is_img2img:
        face_positive_prefix = (
            "RAW photo, photorealistic, 8k uhd, dslr, soft lighting, "
            "high quality, film grain, Fujifilm XT3, "
        )
        if not any(tok in final_prompt.lower() for tok in ["raw photo", "photorealistic", "8k"]):
            final_prompt = face_positive_prefix + final_prompt
        if req.negative_prompt is None or req.negative_prompt.strip() == "":
            final_negative = FACE_NEGATIVE

    try:
        output_type = "latent" if (req.disable_vae and model_choice == "sd15") else "pil"
        target_w = req.width or 512
        target_h = req.height or 512

        if is_img2img:
            raw_b64 = req.image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_data = base64.b64decode(raw_b64)
            init_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            init_img = ImageOps.fit(
                init_img, (target_w, target_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )
            if req.face_preserve:
                init_img = init_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))

            strength = req.strength if req.strength is not None else 0.50
            if req.face_preserve:
                strength = max(0.30, min(strength, 0.60))

            output = img2img_pipe(
                prompt=final_prompt,
                negative_prompt=final_negative,
                image=init_img,
                strength=strength,
                num_inference_steps=max(req.steps, 30),
                guidance_scale=req.cfg_scale,
                generator=generator,
                output_type=output_type
            )
        else:
            output = txt2img_pipe(
                prompt=final_prompt,
                negative_prompt=final_negative,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg_scale,
                width=target_w,
                height=target_h,
                generator=generator,
                output_type=output_type
            )

        if req.disable_vae and model_choice == "sd15":
            image = decode_latents_without_vae(output.images)
        else:
            image = output.images[0]

        elapsed = round(time.time() - start_time, 2)

        filename = f"{model_choice}_{int(time.time())}_{seed}.png"
        filepath = os.path.join(OUTPUTS_DIR, filename)
        image.save(filepath)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "success": True,
            "filename": filename,
            "image_base64": f"data:image/png;base64,{img_str}",
            "elapsed": elapsed,
            "seed": seed,
            "model": model_choice,
            "mode": "img2img" if is_img2img else "txt2img",
            "prompt": final_prompt,
            "face_preserve": req.face_preserve,
            "strength_used": strength if is_img2img else None,
            "vae_disabled": req.disable_vae
        }
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gallery")
def get_gallery():
    images = []
    if os.path.exists(OUTPUTS_DIR):
        for f in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
            if f.endswith((".png", ".jpg", ".webp")):
                images.append(f)
    return {"images": images[:40]}


@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    path = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/api/images/{filename}")
def delete_image(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUTS_DIR, safe_name)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted image: {safe_name}")
        return {"success": True, "deleted": safe_name}
    raise HTTPException(status_code=404, detail="File not found")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
PYEOF
info "Created server.py"

# ── static/index.html ────────────────────────────────────────────────────────
cat > "${STATIC_DIR}/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stable Diffusion Local • SD 1.5 + SDXL</title>
  <meta name="description" content="Local Stable Diffusion — Realistic Vision v5.1 & RealVisXL v4.0. Text-to-image and image-to-image with face preserve mode.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="app-container">
    <header class="top-header">
      <div class="brand">
        <div class="logo-icon">⚡</div>
        <div class="brand-text">
          <h1>Stable Diffusion Local</h1>
          <span class="model-badge">Offline • GPU • SD 1.5 & SDXL</span>
        </div>
      </div>
      <div class="system-status">
        <div class="status-pill offline-pill"><span class="dot green"></span><span>100% Offline</span></div>
        <div class="status-pill gpu-pill"><span class="dot blue"></span><span id="gpuName">GPU</span></div>
        <div class="status-pill vram-pill"><span id="vramFreeText">VRAM: —</span></div>
      </div>
    </header>
    <main class="workspace">
      <section class="controls-panel">
        <div class="panel-header">
          <h2>Generation Parameters</h2>
          <span class="mode-badge" id="modeBadge">Mode: Text-to-Image</span>
        </div>
        <div class="model-selector-group">
          <label class="section-label">🤖 AI Model</label>
          <div class="model-pills">
            <button type="button" class="model-pill active" id="btnModelSd15">
              <div class="pill-top"><span class="pill-title">Realistic Vision v5.1</span><span class="pill-badge sd15-badge">SD 1.5</span></div>
              <span class="pill-sub">Fast (~3s) • 512p • Great faces</span>
              <span class="pill-status" id="sd15Status">● Loaded</span>
            </button>
            <button type="button" class="model-pill" id="btnModelSdxl">
              <div class="pill-top"><span class="pill-title">RealVisXL v4.0 ✨</span><span class="pill-badge sdxl-badge">SDXL</span></div>
              <span class="pill-sub">1024p • Dual CLIP • Highest accuracy</span>
              <span class="pill-status" id="sdxlStatus">⏳ Loading…</span>
            </button>
          </div>
        </div>
        <div class="input-group">
          <label for="prompt">✏️ Prompt</label>
          <textarea id="prompt" rows="3" placeholder="e.g., woman wearing a red bikini on the beach, photorealistic, 8k, cinematic light"></textarea>
        </div>
        <div class="input-group">
          <label for="negPrompt">🚫 Negative Prompt <span class="label-hint">(leave blank for auto face-quality)</span></label>
          <textarea id="negPrompt" rows="2" placeholder="Leave empty for best face/body quality (auto-fills)"></textarea>
        </div>
        <div class="ref-image-card">
          <div class="ref-header">
            <div><div class="ref-title">🖼️ Reference Image (Image-to-Image)</div><div class="ref-desc">Upload a photo — model edits it following your prompt</div></div>
            <button class="btn-text btn-danger" id="btnClearRef" style="display: none;">✕ Remove</button>
          </div>
          <div class="ref-dropzone" id="refDropzone">
            <input type="file" id="refFileInput" accept="image/*" style="display: none;">
            <div class="dropzone-content" id="dropzoneContent"><span class="dropzone-icon">📷</span><span>Click or drag & drop your photo here</span><span class="dropzone-hint">Faces & body shapes are preserved by the model</span></div>
            <img id="refPreviewThumb" src="" alt="Reference Preview" style="display: none;">
          </div>
          <div class="strength-group" id="strengthGroup" style="display: none;">
            <div class="label-val"><label for="strength">Edit Strength (Denoising)</label><span id="strengthVal">0.50</span></div>
            <input type="range" id="strength" min="0.15" max="0.95" step="0.05" value="0.50">
            <div class="strength-hints"><span>0.30 (Keep face exact)</span><span>0.50 (Best balance)</span><span>0.85 (Full redesign)</span></div>
            <div class="face-preserve-row">
              <label class="toggle-label" for="facePreserveToggle">
                <span class="toggle-icon">👤</span>
                <span class="toggle-text"><b>Face & Body Preserve Mode</b><small>Keeps original face structure, skin, body shape — only changes clothes/background/style</small></span>
                <label class="switch"><input type="checkbox" id="facePreserveToggle" checked><span class="slider round"></span></label>
              </label>
            </div>
          </div>
        </div>
        <div class="vae-toggle-card">
          <div class="vae-toggle-header">
            <div><div class="vae-title">Disable VAE Decoder</div><div class="vae-desc">OFF = Full neural decoding (best for faces). ON = Direct latent (fast preview)</div></div>
            <label class="switch"><input type="checkbox" id="disableVaeToggle"><span class="slider round"></span></label>
          </div>
          <div class="vae-badge vae-enabled" id="vaeStateBadge">VAE: ENABLED (Full Photorealistic Details)</div>
        </div>
        <div class="controls-grid">
          <div class="control-item"><div class="label-val"><label for="steps">Sampling Steps</label><span id="stepsVal">30</span></div><input type="range" id="steps" min="10" max="60" value="30"><div class="slider-hints"><span>10 (Fast)</span><span>30 (Quality)</span><span>60 (Max)</span></div></div>
          <div class="control-item"><div class="label-val"><label for="cfg">CFG Scale</label><span id="cfgVal">7.5</span></div><input type="range" id="cfg" min="1.0" max="15.0" step="0.5" value="7.5"><div class="slider-hints"><span>1 (Creative)</span><span>7.5 (Balanced)</span><span>15 (Strict)</span></div></div>
        </div>
        <div class="controls-grid">
          <div class="control-item"><label for="resolution">Dimensions</label><select id="resolution"><option value="512x512" selected>512 × 512 (Standard)</option><option value="512x768">512 × 768 (Portrait)</option><option value="768x512">768 × 512 (Landscape)</option></select></div>
          <div class="control-item"><div class="label-val"><label for="seed">Seed</label><button class="btn-text" id="btnRandomSeed">🎲 Random</button></div><input type="number" id="seed" placeholder="-1 for random" value="-1"></div>
        </div>
        <button id="btnGenerate" class="btn-generate"><span class="btn-icon">✨</span><span class="btn-text-content" id="btnGenerateText">Generate Image</span></button>
      </section>
      <section class="viewport-panel">
        <div class="preview-card" id="previewCard">
          <div class="empty-state" id="emptyState"><div class="empty-icon">🎨</div><h3>Ready to Generate & Edit</h3><p>Enter a prompt or upload a reference photo.<br>Use <b>Face & Body Preserve Mode</b> to keep real faces exact.</p></div>
          <div class="loading-overlay" id="loadingOverlay" style="display: none;"><div class="spinner"></div><p id="loadingStatusText">Generating…</p><span class="loading-subtext" id="loadingSubtext">Running on GPU (Offline)</span></div>
          <div class="image-viewer" id="imageViewer" style="display: none;">
            <img id="resultImage" src="" alt="Generated artwork">
            <div class="image-meta-bar">
              <div class="meta-tags"><span class="tag" id="metaMode">txt2img</span><span class="tag" id="metaTime">⚡ 0.0s</span><span class="tag" id="metaSeed">Seed: —</span><span class="tag" id="metaVae">VAE: Enabled</span><span class="tag" id="metaFace" style="display:none;">👤 Face Preserve</span></div>
              <div class="meta-actions"><button id="btnUseAsRef" class="btn-action secondary">🖼️ Use as Reference</button><a id="btnDownload" download="artwork.png" class="btn-action">💾 Download</a><button id="btnDeleteImage" class="btn-action danger" title="Delete this image">🗑️ Delete</button></div>
            </div>
          </div>
        </div>
        <div class="gallery-section">
          <div class="gallery-header"><h3>Recent Generations</h3><span class="gallery-tip">Click to view • 🗑️ to delete</span><button id="btnRefreshGallery" class="btn-refresh">🔄</button></div>
          <div class="gallery-strip" id="galleryStrip"><span class="gallery-empty-hint">No images generated yet</span></div>
        </div>
      </section>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
HTMLEOF
info "Created static/index.html"

# ── static/app.js ─────────────────────────────────────────────────────────────
# Copy from current working version
cp "${STATIC_DIR}/app.js" "${STATIC_DIR}/app.js.bak" 2>/dev/null || true

cat > "${STATIC_DIR}/app.js" << 'JSEOF'
document.addEventListener("DOMContentLoaded", () => {
  const promptInput       = document.getElementById("prompt");
  const negPromptInput    = document.getElementById("negPrompt");
  const disableVaeToggle  = document.getElementById("disableVaeToggle");
  const vaeStateBadge     = document.getElementById("vaeStateBadge");
  const stepsInput        = document.getElementById("steps");
  const stepsVal          = document.getElementById("stepsVal");
  const cfgInput          = document.getElementById("cfg");
  const cfgVal            = document.getElementById("cfgVal");
  const resolutionSelect  = document.getElementById("resolution");
  const seedInput         = document.getElementById("seed");
  const btnRandomSeed     = document.getElementById("btnRandomSeed");
  const btnGenerate       = document.getElementById("btnGenerate");
  const btnGenerateText   = document.getElementById("btnGenerateText");
  const modeBadge         = document.getElementById("modeBadge");
  const btnModelSd15 = document.getElementById("btnModelSd15");
  const btnModelSdxl = document.getElementById("btnModelSdxl");
  const sd15Status   = document.getElementById("sd15Status");
  const sdxlStatus   = document.getElementById("sdxlStatus");
  let currentModel   = "sd15";
  const refFileInput      = document.getElementById("refFileInput");
  const refDropzone       = document.getElementById("refDropzone");
  const dropzoneContent   = document.getElementById("dropzoneContent");
  const refPreviewThumb   = document.getElementById("refPreviewThumb");
  const btnClearRef       = document.getElementById("btnClearRef");
  const strengthGroup     = document.getElementById("strengthGroup");
  const strengthInput     = document.getElementById("strength");
  const strengthVal       = document.getElementById("strengthVal");
  const facePreserveToggle = document.getElementById("facePreserveToggle");
  const emptyState       = document.getElementById("emptyState");
  const loadingOverlay   = document.getElementById("loadingOverlay");
  const loadingStatusText = document.getElementById("loadingStatusText");
  const loadingSubtext   = document.getElementById("loadingSubtext");
  const imageViewer      = document.getElementById("imageViewer");
  const resultImage      = document.getElementById("resultImage");
  const metaMode         = document.getElementById("metaMode");
  const metaTime         = document.getElementById("metaTime");
  const metaSeed         = document.getElementById("metaSeed");
  const metaVae          = document.getElementById("metaVae");
  const metaFace         = document.getElementById("metaFace");
  const btnUseAsRef      = document.getElementById("btnUseAsRef");
  const btnDownload      = document.getElementById("btnDownload");
  const btnDeleteImage   = document.getElementById("btnDeleteImage");
  const galleryStrip     = document.getElementById("galleryStrip");
  const btnRefreshGallery = document.getElementById("btnRefreshGallery");
  const gpuName          = document.getElementById("gpuName");
  const vramFreeText     = document.getElementById("vramFreeText");
  let currentRefBase64 = null, activeImageSrc = null, currentActiveFilename = null;

  function setModel(modelId) {
    currentModel = modelId;
    if (modelId === "sdxl") {
      btnModelSdxl.classList.add("active"); btnModelSd15.classList.remove("active");
      resolutionSelect.innerHTML = '<option value="1024x1024" selected>1024 × 1024 (Square · SDXL)</option><option value="896x1152">896 × 1152 (Portrait · SDXL)</option><option value="1152x896">1152 × 896 (Landscape · SDXL)</option>';
      disableVaeToggle.checked = false; updateVaeBadge();
    } else {
      btnModelSd15.classList.add("active"); btnModelSdxl.classList.remove("active");
      resolutionSelect.innerHTML = '<option value="512x512" selected>512 × 512 (Standard · SD 1.5)</option><option value="512x768">512 × 768 (Portrait · SD 1.5)</option><option value="768x512">768 × 512 (Landscape · SD 1.5)</option>';
    }
  }
  btnModelSd15.addEventListener("click", () => setModel("sd15"));
  btnModelSdxl.addEventListener("click", () => setModel("sdxl"));
  stepsInput.addEventListener("input", e => stepsVal.textContent = e.target.value);
  cfgInput.addEventListener("input", e => cfgVal.textContent = parseFloat(e.target.value).toFixed(1));
  strengthInput.addEventListener("input", e => strengthVal.textContent = parseFloat(e.target.value).toFixed(2));

  function updateVaeBadge() {
    if (disableVaeToggle.checked) { vaeStateBadge.textContent = "VAE: DISABLED (Direct Latent Preview)"; vaeStateBadge.className = "vae-badge vae-disabled"; }
    else { vaeStateBadge.textContent = "VAE: ENABLED (Full Photorealistic Details)"; vaeStateBadge.className = "vae-badge vae-enabled"; }
  }
  disableVaeToggle.addEventListener("change", updateVaeBadge);
  btnRandomSeed.addEventListener("click", () => { seedInput.value = -1; });

  function setReferenceImage(dataUrl) {
    currentRefBase64 = dataUrl; refPreviewThumb.src = dataUrl; refPreviewThumb.style.display = "block";
    dropzoneContent.style.display = "none"; btnClearRef.style.display = "block"; strengthGroup.style.display = "block";
    modeBadge.textContent = "Mode: Image-to-Image (Edit)"; modeBadge.style.color = "#38bdf8"; modeBadge.style.backgroundColor = "rgba(6,182,212,0.15)";
    btnGenerateText.textContent = "Edit / Transform Image";
    disableVaeToggle.checked = false; updateVaeBadge();
    cfgInput.value = 10.0; cfgVal.textContent = "10.0"; stepsInput.value = 35; stepsVal.textContent = "35";
    strengthInput.value = 0.50; strengthVal.textContent = "0.50";
  }
  function clearReferenceImage() {
    currentRefBase64 = null; refFileInput.value = ""; refPreviewThumb.src = ""; refPreviewThumb.style.display = "none";
    dropzoneContent.style.display = "flex"; btnClearRef.style.display = "none"; strengthGroup.style.display = "none";
    modeBadge.textContent = "Mode: Text-to-Image"; modeBadge.style.color = ""; modeBadge.style.backgroundColor = "";
    btnGenerateText.textContent = "Generate Image";
    cfgInput.value = 7.5; cfgVal.textContent = "7.5"; stepsInput.value = 30; stepsVal.textContent = "30";
  }
  btnClearRef.addEventListener("click", e => { e.stopPropagation(); clearReferenceImage(); });
  refDropzone.addEventListener("click", () => refFileInput.click());
  refFileInput.addEventListener("change", e => { const f = e.target.files[0]; if (f) { const r = new FileReader(); r.onload = ev => setReferenceImage(ev.target.result); r.readAsDataURL(f); } });
  refDropzone.addEventListener("dragover", e => { e.preventDefault(); refDropzone.style.borderColor = "#6366f1"; });
  refDropzone.addEventListener("dragleave", () => { refDropzone.style.borderColor = ""; });
  refDropzone.addEventListener("drop", e => { e.preventDefault(); refDropzone.style.borderColor = ""; const f = e.dataTransfer.files[0]; if (f && f.type.startsWith("image/")) { const r = new FileReader(); r.onload = ev => setReferenceImage(ev.target.result); r.readAsDataURL(f); } });
  btnUseAsRef.addEventListener("click", () => { if (activeImageSrc) { setReferenceImage(activeImageSrc); refDropzone.scrollIntoView({ behavior: "smooth" }); } });

  async function deleteImageFile(filename) {
    if (!filename) return; const ok = confirm('Delete "' + filename + '" permanently?'); if (!ok) return;
    try { const res = await fetch("/api/images/" + encodeURIComponent(filename), { method: "DELETE" }); if (!res.ok) throw new Error("Failed");
      if (currentActiveFilename === filename) { currentActiveFilename = null; activeImageSrc = null; imageViewer.style.display = "none"; emptyState.style.display = "flex"; }
      await loadGallery();
    } catch (err) { alert("Error: " + err.message); }
  }
  btnDeleteImage.addEventListener("click", () => { if (currentActiveFilename) deleteImageFile(currentActiveFilename); });

  async function checkStatus() {
    try { const res = await fetch("/api/status"); const d = await res.json();
      if (d.device) gpuName.textContent = d.device + " (" + (d.vram_total || "24GB") + ")";
      if (d.vram_free) vramFreeText.textContent = "VRAM Free: " + d.vram_free;
      if (d.sd15_loaded) { sd15Status.textContent = "● Loaded"; sd15Status.style.color = "#34d399"; }
      else if (d.sd15_ready) { sd15Status.textContent = "○ Ready"; sd15Status.style.color = "#fbbf24"; }
      else { sd15Status.textContent = "✕ Missing"; sd15Status.style.color = "#f87171"; }
      if (d.sdxl_loaded) { sdxlStatus.textContent = "● Loaded"; sdxlStatus.style.color = "#34d399"; }
      else if (d.sdxl_download_complete) { sdxlStatus.textContent = "○ Ready"; sdxlStatus.style.color = "#fbbf24"; }
      else if (d.sdxl_ready) { sdxlStatus.textContent = "⬇ " + d.sdxl_download_gb + "/6.6 GB"; sdxlStatus.style.color = "#60a5fa"; }
      else { sdxlStatus.textContent = "✕ Not downloaded"; sdxlStatus.style.color = "#f87171"; }
    } catch (err) { console.warn("Status check pending...", err); }
  }
  checkStatus(); setInterval(checkStatus, 8000);

  async function loadGallery() {
    try { const res = await fetch("/api/gallery"); const d = await res.json(); galleryStrip.innerHTML = "";
      if (d.images && d.images.length > 0) { d.images.forEach(f => {
        const item = document.createElement("div"); item.className = "gallery-item";
        const img = document.createElement("img"); img.src = "/outputs/" + f; img.className = "gallery-thumb"; img.title = f;
        img.addEventListener("click", () => { currentActiveFilename = f; activeImageSrc = "/outputs/" + f; resultImage.src = "/outputs/" + f; btnDownload.href = "/outputs/" + f; btnDownload.download = f; emptyState.style.display = "none"; imageViewer.style.display = "flex"; metaMode.textContent = f.startsWith("sdxl") ? "[SDXL]" : "[SD 1.5]"; metaTime.textContent = "📁 Gallery"; metaSeed.textContent = f; metaVae.textContent = "Offline"; metaFace.style.display = "none"; });
        const del = document.createElement("button"); del.className = "gallery-delete-btn"; del.title = "Delete"; del.textContent = "🗑️";
        del.addEventListener("click", e => { e.stopPropagation(); deleteImageFile(f); });
        item.appendChild(img); item.appendChild(del); galleryStrip.appendChild(item);
      }); } else { const s = document.createElement("span"); s.className = "gallery-empty-hint"; s.textContent = "No images generated yet"; galleryStrip.appendChild(s); }
    } catch (err) { console.warn("Gallery error", err); }
  }
  loadGallery(); btnRefreshGallery.addEventListener("click", loadGallery);

  btnGenerate.addEventListener("click", async () => {
    const prompt = promptInput.value.trim(); if (!prompt) { alert("Please enter a prompt."); promptInput.focus(); return; }
    const [w, h] = resolutionSelect.value.split("x").map(Number);
    const isImg2Img = !!currentRefBase64; const facePreserve = isImg2Img && facePreserveToggle.checked;
    btnGenerate.disabled = true; emptyState.style.display = "none"; imageViewer.style.display = "none"; loadingOverlay.style.display = "flex";
    const mn = currentModel === "sdxl" ? "RealVisXL v4.0 (SDXL)" : "Realistic Vision v5.1";
    loadingStatusText.textContent = (isImg2Img ? "Editing" : "Generating") + " with " + mn + "…";
    try {
      const payload = { model_type: currentModel, prompt, negative_prompt: negPromptInput.value.trim() || null, steps: parseInt(stepsInput.value), cfg_scale: parseFloat(cfgInput.value), seed: parseInt(seedInput.value) || -1, width: w, height: h, disable_vae: disableVaeToggle.checked, image_base64: currentRefBase64 || null, strength: isImg2Img ? parseFloat(strengthInput.value) : undefined, face_preserve: facePreserve };
      const response = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (!response.ok) { let msg = "Generation failed"; try { const t = await response.text(); const j = JSON.parse(t); msg = j.detail || msg; } catch(_) { msg = "Server error (" + response.status + "). Check if model is loaded."; } throw new Error(msg); }
      const result = await response.json(); currentActiveFilename = result.filename; activeImageSrc = result.image_base64; resultImage.src = result.image_base64;
      metaMode.textContent = "[" + result.model.toUpperCase() + "] " + result.mode; metaTime.textContent = "⚡ " + result.elapsed + "s"; metaSeed.textContent = "Seed: " + result.seed;
      metaVae.textContent = result.vae_disabled ? "VAE: Disabled" : "VAE: Enabled";
      if (result.face_preserve) { metaFace.style.display = "inline-block"; metaFace.textContent = "👤 Face Preserve (str=" + result.strength_used + ")"; } else { metaFace.style.display = "none"; }
      btnDownload.href = result.image_base64; btnDownload.download = result.filename; imageViewer.style.display = "flex"; loadGallery();
    } catch (err) { alert("Error: " + err.message); emptyState.style.display = "flex"; } finally { loadingOverlay.style.display = "none"; btnGenerate.disabled = false; }
  });
});
JSEOF
info "Created static/app.js"

# ── static/style.css ──────────────────────────────────────────────────────────
# Copy from existing if available, otherwise use the working version
if [ -f "${STATIC_DIR}/style.css" ] && [ -s "${STATIC_DIR}/style.css" ]; then
    info "style.css already exists, keeping"
else
    info "Creating style.css..."
fi

# Always write the full CSS to ensure consistency
cat > "${STATIC_DIR}/style.css" << 'CSSEOF'
:root{--bg-main:#0b0f19;--bg-card:#111827;--bg-panel:#161f30;--bg-input:#0e1524;--border-color:rgba(255,255,255,.08);--border-active:rgba(99,102,241,.4);--text-main:#f3f4f6;--text-muted:#9ca3af;--primary:#6366f1;--primary-hover:#4f46e5;--accent-cyan:#06b6d4;--accent-green:#10b981;--accent-amber:#f59e0b;--accent-red:#ef4444;--radius:12px;--shadow:0 10px 25px -5px rgba(0,0,0,.5),0 8px 10px -6px rgba(0,0,0,.5);--font:'Inter',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--font-mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background-color:var(--bg-main);color:var(--text-main);font-family:var(--font);min-height:100vh;display:flex;flex-direction:column;overflow-x:hidden}
.app-container{display:flex;flex-direction:column;height:100vh}
.top-header{height:64px;background-color:var(--bg-card);border-bottom:1px solid var(--border-color);display:flex;align-items:center;justify-content:space-between;padding:0 24px}
.brand{display:flex;align-items:center;gap:12px}
.logo-icon{width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:18px}
.brand-text h1{font-size:17px;font-weight:700;letter-spacing:-.02em}
.model-badge{font-size:11px;color:var(--accent-cyan);font-weight:500;letter-spacing:.05em;text-transform:uppercase}
.system-status{display:flex;gap:12px}
.status-pill{display:flex;align-items:center;gap:8px;background-color:var(--bg-input);border:1px solid var(--border-color);padding:6px 14px;border-radius:20px;font-size:12px;font-weight:500}
.dot{width:8px;height:8px;border-radius:50%}
.dot.green{background-color:var(--accent-green);box-shadow:0 0 8px var(--accent-green)}
.dot.blue{background-color:var(--accent-cyan);box-shadow:0 0 8px var(--accent-cyan)}
.workspace{display:grid;grid-template-columns:460px 1fr;flex:1;overflow:hidden}
.controls-panel{background-color:var(--bg-card);border-right:1px solid var(--border-color);padding:24px;overflow-y:auto;display:flex;flex-direction:column;gap:18px}
.panel-header{display:flex;justify-content:space-between;align-items:center}
.panel-header h2{font-size:14px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em}
.mode-badge{font-size:11px;font-family:var(--font-mono);font-weight:600;background-color:rgba(99,102,241,.15);color:#a5b4fc;padding:4px 8px;border-radius:6px}
.section-label{font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em}
.model-selector-group{display:flex;flex-direction:column;gap:8px}
.model-pills{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.model-pill{background-color:var(--bg-input);border:1px solid var(--border-color);border-radius:var(--radius);padding:10px 12px;text-align:left;cursor:pointer;transition:all .2s ease;display:flex;flex-direction:column;gap:4px}
.model-pill:hover{border-color:rgba(99,102,241,.5);background-color:rgba(99,102,241,.05)}
.model-pill.active{border-color:var(--primary);background:linear-gradient(135deg,rgba(99,102,241,.15),rgba(6,182,212,.1));box-shadow:0 0 14px rgba(99,102,241,.3)}
.pill-top{display:flex;align-items:center;justify-content:space-between;gap:6px}
.pill-title{font-size:13px;font-weight:600;color:#e2e8f0}
.model-pill.active .pill-title{color:#c7d2fe}
.pill-badge{font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;letter-spacing:.05em;text-transform:uppercase;flex-shrink:0}
.sd15-badge{background:rgba(99,102,241,.25);color:#a5b4fc;border:1px solid rgba(99,102,241,.4)}
.sdxl-badge{background:rgba(245,158,11,.2);color:#fcd34d;border:1px solid rgba(245,158,11,.4)}
.pill-sub{font-size:10px;color:var(--text-muted);font-family:var(--font-mono)}
.model-pill.active .pill-sub{color:var(--accent-cyan)}
.pill-status{font-size:10px;font-family:var(--font-mono);font-weight:600;color:#34d399;margin-top:2px}
.input-group{display:flex;flex-direction:column;gap:6px}
label{font-size:13px;font-weight:500;color:var(--text-muted)}
.label-hint{font-size:10px;font-weight:400;color:#64748b;font-style:italic}
textarea,select,input[type="number"]{background-color:var(--bg-input);border:1px solid var(--border-color);border-radius:var(--radius);color:var(--text-main);font-family:var(--font);font-size:13px;padding:10px 14px;outline:none;transition:border-color .2s,box-shadow .2s;resize:vertical}
textarea:focus,select:focus,input[type="number"]:focus{border-color:var(--primary);box-shadow:0 0 0 2px rgba(99,102,241,.2)}
.ref-image-card{background:linear-gradient(135deg,rgba(6,182,212,.06),rgba(99,102,241,.06));border:1px solid rgba(6,182,212,.25);border-radius:var(--radius);padding:14px;display:flex;flex-direction:column;gap:10px}
.ref-header{display:flex;justify-content:space-between;align-items:flex-start}
.ref-title{font-size:13px;font-weight:600;color:#a5f3fc}
.ref-desc{font-size:11px;color:var(--text-muted)}
.ref-dropzone{border:2px dashed rgba(6,182,212,.3);border-radius:8px;padding:14px;text-align:center;cursor:pointer;background-color:rgba(14,21,36,.5);transition:border-color .2s,background-color .2s;position:relative;min-height:80px;display:flex;align-items:center;justify-content:center}
.ref-dropzone:hover{border-color:var(--accent-cyan);background-color:rgba(6,182,212,.08)}
.dropzone-content{display:flex;flex-direction:column;align-items:center;gap:4px;font-size:12px;color:var(--text-muted)}
.dropzone-icon{font-size:24px}
.dropzone-hint{font-size:10px;color:#64748b}
#refPreviewThumb{max-height:120px;max-width:100%;border-radius:6px;object-fit:contain;box-shadow:0 4px 10px rgba(0,0,0,.4)}
.strength-group{display:flex;flex-direction:column;gap:6px;margin-top:4px;padding-top:8px;border-top:1px solid rgba(255,255,255,.06)}
.strength-hints,.slider-hints{display:flex;justify-content:space-between;font-size:10px;color:#64748b;font-family:var(--font-mono)}
.slider-hints{font-size:9px;color:#4b5563;margin-top:-2px}
.face-preserve-row{margin-top:8px;padding:10px 12px;background:linear-gradient(135deg,rgba(16,185,129,.08),rgba(6,182,212,.06));border:1px solid rgba(16,185,129,.25);border-radius:8px}
.toggle-label{display:flex;align-items:center;gap:10px;cursor:pointer;color:var(--text-main);font-size:13px;font-weight:400}
.toggle-icon{font-size:20px;flex-shrink:0}
.toggle-text{flex:1;display:flex;flex-direction:column;gap:2px}
.toggle-text b{font-size:12px;font-weight:600;color:#a7f3d0}
.toggle-text small{font-size:10px;color:var(--text-muted);line-height:1.4}
.vae-toggle-card{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(6,182,212,.08));border:1px solid rgba(99,102,241,.25);border-radius:var(--radius);padding:14px;display:flex;flex-direction:column;gap:10px}
.vae-toggle-header{display:flex;justify-content:space-between;align-items:center}
.vae-title{font-size:14px;font-weight:600;color:#c7d2fe}
.vae-desc{font-size:11px;color:var(--text-muted)}
.vae-badge{font-size:11px;font-family:var(--font-mono);font-weight:600;padding:4px 8px;border-radius:6px;text-align:center;letter-spacing:.04em;transition:background-color .3s,color .3s}
.vae-badge.vae-enabled{background-color:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.25)}
.vae-badge.vae-disabled{background-color:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.25)}
.switch{position:relative;display:inline-block;width:44px;height:24px}
.switch input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background-color:#374151;transition:.3s;border-radius:24px}
.slider:before{position:absolute;content:"";height:18px;width:18px;left:3px;bottom:3px;background-color:white;transition:.3s;border-radius:50%}
input:checked+.slider{background:linear-gradient(135deg,#6366f1,#06b6d4)}
input:checked+.slider:before{transform:translateX(20px)}
.controls-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.control-item{display:flex;flex-direction:column;gap:6px}
.label-val{display:flex;justify-content:space-between;align-items:center}
.label-val span{font-family:var(--font-mono);font-size:12px;color:var(--accent-cyan)}
input[type="range"]{accent-color:var(--primary);cursor:pointer}
.btn-text{background:none;border:none;color:var(--accent-cyan);font-size:11px;cursor:pointer;padding:2px 4px}
.btn-danger{color:var(--accent-red)}
.btn-generate{margin-top:6px;background:linear-gradient(135deg,#6366f1,#8b5cf6);border:none;border-radius:var(--radius);color:#fff;padding:14px;font-size:15px;font-weight:600;font-family:var(--font);cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;box-shadow:0 4px 15px rgba(99,102,241,.4);transition:transform .15s,box-shadow .15s}
.btn-generate:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 20px rgba(99,102,241,.5)}
.btn-generate:disabled{opacity:.6;cursor:not-allowed}
.viewport-panel{display:flex;flex-direction:column;padding:24px;gap:20px;overflow-y:auto}
.preview-card{flex:1;min-height:480px;background-color:var(--bg-card);border:1px solid var(--border-color);border-radius:var(--radius);position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;box-shadow:var(--shadow)}
.empty-state{text-align:center;padding:40px;max-width:400px}
.empty-icon{font-size:48px;margin-bottom:12px}
.empty-state h3{font-size:18px;margin-bottom:6px}
.empty-state p{font-size:13px;color:var(--text-muted);line-height:1.5}
.loading-overlay{position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(11,15,25,.85);backdrop-filter:blur(8px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;z-index:10}
.spinner{width:50px;height:50px;border:4px solid rgba(99,102,241,.2);border-top-color:var(--primary);border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-subtext{font-size:12px;color:var(--accent-cyan);font-family:var(--font-mono)}
.image-viewer{display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%;position:relative}
.image-viewer img{max-width:90%;max-height:70vh;object-fit:contain;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.6)}
.image-meta-bar{display:flex;align-items:center;justify-content:space-between;width:90%;margin-top:14px;flex-wrap:wrap;gap:8px}
.meta-tags{display:flex;gap:8px;flex-wrap:wrap}
.tag{font-family:var(--font-mono);font-size:11px;background-color:var(--bg-input);border:1px solid var(--border-color);padding:4px 10px;border-radius:6px;color:var(--text-muted)}
.meta-actions{display:flex;gap:10px}
.btn-action{background-color:var(--primary);color:#fff;border:none;text-decoration:none;padding:6px 14px;border-radius:6px;font-size:12px;font-weight:500;cursor:pointer;transition:background-color .2s;display:flex;align-items:center;gap:6px}
.btn-action:hover{background-color:var(--primary-hover)}
.btn-action.secondary{background-color:rgba(6,182,212,.15);color:#38bdf8;border:1px solid rgba(6,182,212,.3)}
.btn-action.secondary:hover{background-color:rgba(6,182,212,.25)}
.btn-action.danger{background-color:rgba(239,68,68,.15);color:#f87171;border:1px solid rgba(239,68,68,.3)}
.btn-action.danger:hover{background-color:rgba(239,68,68,.3);color:#fca5a5}
.gallery-section{background-color:var(--bg-card);border:1px solid var(--border-color);border-radius:var(--radius);padding:16px}
.gallery-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.gallery-header h3{font-size:13px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em}
.gallery-tip{font-size:11px;color:#64748b;flex:1}
.btn-refresh{background:none;border:none;cursor:pointer;font-size:14px;color:var(--text-muted)}
.gallery-strip{display:flex;gap:12px;overflow-x:auto;padding:6px 2px 10px 2px}
.gallery-item{position:relative;display:inline-block;flex-shrink:0}
.gallery-thumb{width:80px;height:80px;border-radius:6px;object-fit:cover;border:2px solid transparent;cursor:pointer;transition:transform .15s,border-color .15s;display:block}
.gallery-item:hover .gallery-thumb{transform:scale(1.05);border-color:var(--primary)}
.gallery-delete-btn{position:absolute;top:4px;right:4px;width:22px;height:22px;border-radius:4px;background:rgba(15,23,42,.9);border:1px solid rgba(239,68,68,.5);color:#f87171;font-size:11px;display:flex;align-items:center;justify-content:center;cursor:pointer;opacity:0;transition:opacity .2s,background .2s;z-index:2}
.gallery-item:hover .gallery-delete-btn{opacity:1}
.gallery-delete-btn:hover{background:#ef4444;color:#fff}
.gallery-empty-hint{font-size:12px;color:var(--text-muted)}
CSSEOF
info "Created static/style.css"

success "All application source files created"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Create .gitignore
# ══════════════════════════════════════════════════════════════════════════════
step "Step 7/9: Creating .gitignore"

cat > "${PROJECT_DIR}/.gitignore" << 'GIEOF'
# Virtual Environments
venv/
.venv/
env/

# Generated Images and Outputs
outputs/
*.png
*.jpg
*.jpeg
*.webp

# Heavy Model Checkpoints
models/*.safetensors
models/*.ckpt
models/*.bin
models/*.pt

# Python bytecode & Caches
__pycache__/
*.py[cod]
.cache/
.huggingface/

# Runtime Logs
*.log
nohup.out

# IDE & OS Metadata
.DS_Store
.vscode/
.idea/
GIEOF

success "Created .gitignore"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: Create run script and desktop shortcut
# ══════════════════════════════════════════════════════════════════════════════
step "Step 8/9: Creating run script and desktop shortcut"

# run_sd.sh
cat > "${PROJECT_DIR}/run_sd.sh" << RUNEOF
#!/usr/bin/env bash
set -e
DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"
cd "\$DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "══════════════════════════════════════════════════"
echo " Starting Stable Diffusion (SD 1.5 + SDXL)"
echo " Mode: 100% Offline (127.0.0.1:${SERVER_PORT})"
echo "══════════════════════════════════════════════════"

source "\$DIR/venv/bin/activate"

( sleep 3
  if which xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:${SERVER_PORT}" >/dev/null 2>&1 &
  elif which google-chrome >/dev/null 2>&1; then
    google-chrome "http://127.0.0.1:${SERVER_PORT}" >/dev/null 2>&1 &
  elif which firefox >/dev/null 2>&1; then
    firefox "http://127.0.0.1:${SERVER_PORT}" >/dev/null 2>&1 &
  fi
) &

exec python3 server.py
RUNEOF
chmod +x "${PROJECT_DIR}/run_sd.sh"
info "Created run_sd.sh"

# Desktop shortcut
cat > "${HOME}/Desktop/Stable_Diffusion.desktop" << DESKEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Stable Diffusion (SD 1.5 + SDXL)
Comment=Local Offline AI Image Generator — Realistic Vision v5.1 & RealVisXL v4.0
Exec=${PROJECT_DIR}/run_sd.sh
Icon=image-x-generic
Terminal=true
Categories=Graphics;ArtificialIntelligence;
StartupNotify=true
DESKEOF
chmod +x "${HOME}/Desktop/Stable_Diffusion.desktop" 2>/dev/null || true
info "Created desktop shortcut"

success "Run script and desktop shortcut ready"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9: Initialize git and start the server
# ══════════════════════════════════════════════════════════════════════════════
step "Step 9/9: Initializing git repo and starting server"

cd "${PROJECT_DIR}"
if [ ! -d ".git" ]; then
    git init
    git add -A
    git commit -m "Initial commit: Stable Diffusion Local (SD 1.5 + SDXL) with face preserve mode"
    info "Git repo initialized with initial commit"
else
    git add -A
    git diff --cached --quiet || git commit -m "Update: Stable Diffusion Local setup" || true
    info "Git repo updated"
fi

# Kill any existing server on the port
fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true
sleep 1

# Start the server
info "Starting server on http://127.0.0.1:${SERVER_PORT} ..."
source "${VENV_DIR}/bin/activate"
nohup python3 server.py >> server.log 2>&1 &
SERVER_PID=$!
sleep 8

# Verify server is running
if curl -s "http://127.0.0.1:${SERVER_PORT}/api/status" >/dev/null 2>&1; then
    success "Server is running! PID: ${SERVER_PID}"
else
    warn "Server may still be loading models. Check server.log for details."
fi

echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ SETUP COMPLETE!${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}🌐 Open:${NC}     http://127.0.0.1:${SERVER_PORT}"
echo -e "  ${CYAN}📁 Project:${NC}  ${PROJECT_DIR}"
echo -e "  ${CYAN}🚀 Run:${NC}      ${PROJECT_DIR}/run_sd.sh"
echo -e "  ${CYAN}🖥️ Desktop:${NC}  Double-click 'Stable Diffusion' on Desktop"
echo ""
echo -e "  ${YELLOW}Models installed:${NC}"
echo -e "    • Realistic Vision v5.1 (SD 1.5) — 512p, ~3s generation"
echo -e "    • RealVisXL v4.0 (SDXL) — 1024p, highest accuracy"
echo ""
echo -e "  ${YELLOW}Features:${NC}"
echo -e "    • Text-to-Image & Image-to-Image"
echo -e "    • Face & Body Preserve Mode"
echo -e "    • VAE disable toggle"
echo -e "    • 100% offline after setup"
echo ""
