#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Stable Diffusion Local — Clone & Run (Automated Setup)
#
#  Clones from: https://github.com/nadunMadhushanka/test-sdf.git
#
#  What this script does:
#    1. Clones the project from GitHub
#    2. Creates Python virtual environment
#    3. Installs PyTorch (CUDA 12.1) + diffusers + transformers + FastAPI
#    4. Downloads Realistic Vision v5.1 (SD 1.5, ~2 GB)
#    5. Downloads RealVisXL v4.0 (SDXL, ~6.5 GB)
#    6. Creates SDXL HuggingFace cache configs (offline compatibility)
#    7. Creates desktop shortcut
#    8. Starts the server
#
#  Usage (run this ONE command on any fresh machine with an NVIDIA GPU):
#    curl -sL https://raw.githubusercontent.com/nadunMadhushanka/test-sdf/main/setup_stable_diffusion.sh | bash
#
#  Or manually:
#    chmod +x setup_stable_diffusion.sh
#    ./setup_stable_diffusion.sh
#
#  Requirements:
#    - Ubuntu/Debian Linux with NVIDIA GPU + drivers
#    - Python 3.10+, git, curl
#    - Internet connection (for initial setup only — runs fully offline after)
#    - ~12 GB free disk space
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_URL="https://github.com/nadunMadhushanka/test-sdf.git"
PROJECT_DIR="${HOME}/Desktop/stable-diffusion"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/models"
SERVER_PORT=7860

# Model URLs
SD15_URL="https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
SD15_FILE="Realistic_Vision_V5.1.safetensors"
SDXL_URL="https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors"
SDXL_FILE="RealVisXL_V4.0.safetensors"

# SDXL cache path
SDXL_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[  OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
step()    { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }

# ══════════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║   Stable Diffusion Local — Automated Setup           ║"
echo "  ║   SD 1.5 (Realistic Vision) + SDXL (RealVisXL)      ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════════
step "Step 1/8: Pre-flight checks"

command -v nvidia-smi &>/dev/null || error "nvidia-smi not found. Install NVIDIA drivers first."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
info "GPU: ${GPU_NAME} (${GPU_VRAM})"

command -v python3 &>/dev/null || error "python3 not found. Install Python 3.10+."
info "Python: $(python3 --version 2>&1)"

command -v git &>/dev/null || { warn "Installing git..."; sudo apt-get update -qq && sudo apt-get install -y -qq git; }
command -v curl &>/dev/null || { warn "Installing curl..."; sudo apt-get update -qq && sudo apt-get install -y -qq curl; }

# Ensure python3-venv is available
python3 -m venv --help &>/dev/null || { warn "Installing python3-venv..."; sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv; }

AVAIL_GB=$(df -BG "${HOME}" | awk 'NR==2 {print $4}' | tr -d 'G')
info "Disk space: ${AVAIL_GB} GB available"
[ "${AVAIL_GB}" -lt 12 ] && warn "Low disk space! Need ~12 GB."

success "Pre-flight checks passed"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Clone the project
# ══════════════════════════════════════════════════════════════════════════════
step "Step 2/8: Cloning project from GitHub"

if [ -d "${PROJECT_DIR}/.git" ]; then
    info "Project already exists, pulling latest..."
    cd "${PROJECT_DIR}"
    git pull origin main 2>/dev/null || git pull 2>/dev/null || warn "Pull failed, using existing code"
else
    if [ -d "${PROJECT_DIR}" ]; then
        # Directory exists but not a git repo — back up and clone fresh
        warn "Directory exists but is not a git repo. Backing up..."
        mv "${PROJECT_DIR}" "${PROJECT_DIR}.bak.$(date +%s)"
    fi
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
fi

mkdir -p "${MODELS_DIR}" "${PROJECT_DIR}/outputs"
success "Project cloned to ${PROJECT_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Create Python virtual environment
# ══════════════════════════════════════════════════════════════════════════════
step "Step 3/8: Setting up Python virtual environment"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    info "Created new venv"
else
    info "Existing venv found, reusing"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q
success "Virtual environment ready"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Install Python packages
# ══════════════════════════════════════════════════════════════════════════════
step "Step 4/8: Installing Python packages"

info "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

info "Installing diffusers, transformers, accelerate..."
pip install diffusers transformers accelerate -q

info "Installing FastAPI + Uvicorn + Pillow..."
pip install fastapi uvicorn pillow numpy pydantic python-multipart -q

# Verify
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available(): print(f'  GPU: {torch.cuda.get_device_name(0)}')
" || error "PyTorch installation failed"

success "All packages installed"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Download AI models
# ══════════════════════════════════════════════════════════════════════════════
step "Step 5/8: Downloading AI models"

download_model() {
    local url="$1" dest="$2" name="$3"
    if [ -f "${dest}" ]; then
        local size
        size=$(stat -c%s "${dest}" 2>/dev/null || echo 0)
        if [ "${size}" -gt 1000000000 ]; then
            info "${name} already exists ($(echo "scale=2; ${size}/1073741824" | bc) GB), skipping"
            return 0
        fi
    fi
    info "Downloading ${name}... (this may take 5-15 minutes)"
    curl -L -C - --retry 5 --progress-bar -o "${dest}" "${url}" || \
    curl -L --retry 10 --progress-bar -o "${dest}" "${url}" || \
    error "Failed to download ${name}"
    success "${name} downloaded"
}

download_model "${SD15_URL}" "${MODELS_DIR}/${SD15_FILE}" "Realistic Vision v5.1 (SD 1.5, ~2 GB)"
download_model "${SDXL_URL}" "${MODELS_DIR}/${SDXL_FILE}" "RealVisXL v4.0 (SDXL, ~6.5 GB)"

success "All models downloaded"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Create SDXL offline configs
# ══════════════════════════════════════════════════════════════════════════════
step "Step 6/8: Creating SDXL offline config files"

mkdir -p "${SDXL_CACHE_DIR}"/{scheduler,unet,vae,text_encoder,text_encoder_2}

cat > "${SDXL_CACHE_DIR}/model_index.json" << 'EOF'
{"_class_name":"StableDiffusionXLPipeline","_diffusers_version":"0.25.0","force_zeros_for_empty_prompt":true,"add_watermarker":false,"scheduler":["diffusers","EulerDiscreteScheduler"],"text_encoder":["transformers","CLIPTextModel"],"text_encoder_2":["transformers","CLIPTextModelWithProjection"],"tokenizer":["transformers","CLIPTokenizer"],"tokenizer_2":["transformers","CLIPTokenizer"],"unet":["diffusers","UNet2DConditionModel"],"vae":["diffusers","AutoencoderKL"]}
EOF

cat > "${SDXL_CACHE_DIR}/scheduler/scheduler_config.json" << 'EOF'
{"_class_name":"EulerDiscreteScheduler","_diffusers_version":"0.25.0","beta_end":0.012,"beta_schedule":"scaled_linear","beta_start":0.00085,"interpolation_type":"linear","num_train_timesteps":1000,"prediction_type":"epsilon","sample_max_value":1.0,"set_alpha_to_one":false,"skip_prk_steps":true,"steps_offset":1,"timestep_spacing":"leading","use_karras_sigmas":false}
EOF

cat > "${SDXL_CACHE_DIR}/unet/config.json" << 'EOF'
{"_class_name":"UNet2DConditionModel","_diffusers_version":"0.25.0","act_fn":"silu","addition_embed_type":"text_time","addition_embed_type_num_heads":64,"addition_time_embed_dim":256,"attention_head_dim":[5,10,20],"block_out_channels":[320,640,1280],"center_input_sample":false,"cross_attention_dim":2048,"down_block_types":["DownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D"],"downsample_padding":1,"in_channels":4,"layers_per_block":2,"mid_block_type":"UNetMidBlock2DCrossAttn","norm_eps":1e-05,"norm_num_groups":32,"out_channels":4,"projection_class_embeddings_input_dim":2816,"sample_size":128,"transformer_layers_per_block":[1,2,10],"up_block_types":["CrossAttnUpBlock2D","CrossAttnUpBlock2D","UpBlock2D"],"use_linear_projection":true}
EOF

cat > "${SDXL_CACHE_DIR}/vae/config.json" << 'EOF'
{"_class_name":"AutoencoderKL","_diffusers_version":"0.25.0","act_fn":"silu","block_out_channels":[128,256,512,512],"down_block_types":["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],"force_upcast":true,"in_channels":3,"latent_channels":4,"layers_per_block":2,"norm_num_groups":32,"out_channels":3,"sample_size":1024,"scaling_factor":0.13025,"up_block_types":["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"]}
EOF

cat > "${SDXL_CACHE_DIR}/text_encoder/config.json" << 'EOF'
{"_class_name":"CLIPTextModel","architectures":["CLIPTextModel"],"hidden_act":"quick_gelu","hidden_size":768,"intermediate_size":3072,"layer_norm_eps":1e-05,"max_position_embeddings":77,"model_type":"clip_text_model","num_attention_heads":12,"num_hidden_layers":12,"projection_dim":768,"vocab_size":49408}
EOF

cat > "${SDXL_CACHE_DIR}/text_encoder_2/config.json" << 'EOF'
{"_class_name":"CLIPTextModelWithProjection","architectures":["CLIPTextModelWithProjection"],"hidden_act":"gelu","hidden_size":1280,"intermediate_size":5120,"layer_norm_eps":1e-05,"max_position_embeddings":77,"model_type":"clip_text_model","num_attention_heads":20,"num_hidden_layers":32,"projection_dim":1280,"vocab_size":49408}
EOF

success "SDXL offline configs created"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Create desktop shortcut
# ══════════════════════════════════════════════════════════════════════════════
step "Step 7/8: Creating desktop shortcut"

chmod +x "${PROJECT_DIR}/run_sd.sh"

cat > "${HOME}/Desktop/Stable_Diffusion.desktop" << DEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Stable Diffusion (SD 1.5 + SDXL)
Comment=Local Offline AI Image Generator
Exec=${PROJECT_DIR}/run_sd.sh
Icon=image-x-generic
Terminal=true
Categories=Graphics;ArtificialIntelligence;
StartupNotify=true
DEOF
chmod +x "${HOME}/Desktop/Stable_Diffusion.desktop" 2>/dev/null || true

success "Desktop shortcut created"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8: Start the server
# ══════════════════════════════════════════════════════════════════════════════
step "Step 8/8: Starting server"

cd "${PROJECT_DIR}"
fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true
sleep 1

source "${VENV_DIR}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

info "Starting server on http://127.0.0.1:${SERVER_PORT} ..."
nohup python3 server.py >> server.log 2>&1 &
SERVER_PID=$!
sleep 8

if curl -s "http://127.0.0.1:${SERVER_PORT}/api/status" >/dev/null 2>&1; then
    success "Server is running! (PID: ${SERVER_PID})"
else
    warn "Server is loading models... check server.log"
fi

# Open browser
( sleep 3
  if command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:${SERVER_PORT}" &>/dev/null &
  elif command -v google-chrome &>/dev/null; then
    google-chrome "http://127.0.0.1:${SERVER_PORT}" &>/dev/null &
  fi
) &

echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ SETUP COMPLETE!${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}🌐 Open in browser:${NC}  http://127.0.0.1:${SERVER_PORT}"
echo -e "  ${CYAN}📁 Project:${NC}          ${PROJECT_DIR}"
echo -e "  ${CYAN}🚀 Run again:${NC}        ${PROJECT_DIR}/run_sd.sh"
echo -e "  ${CYAN}🖥️  Desktop:${NC}          Double-click 'Stable Diffusion' on Desktop"
echo -e "  ${CYAN}📦 GitHub:${NC}           ${REPO_URL}"
echo ""
echo -e "  ${YELLOW}Models:${NC}"
echo -e "    • Realistic Vision v5.1 (SD 1.5) — 512p, ~3s"
echo -e "    • RealVisXL v4.0 (SDXL)          — 1024p, highest accuracy"
echo ""
echo -e "  ${YELLOW}Features:${NC}"
echo -e "    • Text-to-Image & Image-to-Image"
echo -e "    • Face & Body Preserve Mode"
echo -e "    • VAE toggle, gallery, seed control"
echo -e "    • 100% offline after this setup"
echo ""
echo -e "  ${CYAN}To run on a new machine:${NC}"
echo -e "    curl -sL https://raw.githubusercontent.com/nadunMadhushanka/test-sdf/main/setup_stable_diffusion.sh | bash"
echo ""
