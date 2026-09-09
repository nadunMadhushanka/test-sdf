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
    model_type: Optional[str] = "sd15"        # "sd15" or "sdxl"
    prompt: str
    negative_prompt: Optional[str] = None      # if None → FACE_NEGATIVE is used
    steps: Optional[int] = 30
    cfg_scale: Optional[float] = 7.5
    seed: Optional[int] = -1
    width: Optional[int] = 512
    height: Optional[int] = 512
    disable_vae: Optional[bool] = False
    image_base64: Optional[str] = None
    strength: Optional[float] = 0.50          # lower = preserves more of original face/body
    # Face-preserve mode: injects extra face-positive tokens into prompt + smart negative
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
        sdxl_complete = size_bytes > 6_400_000_000  # ~6.4+ GB = downloaded
    return {
        "models": [
            {
                "id": "sd15",
                "name": "Realistic Vision v5.1 (SD 1.5)",
                "resolution": "512 × 512",
                "available": os.path.exists(SD15_MODEL_PATH),
                "loaded": pipelines["sd15"]["txt2img"] is not None,
                "description": "Fast generation (~3s), portrait-optimized"
            },
            {
                "id": "sdxl",
                "name": "RealVisXL v4.0 (SDXL)",
                "resolution": "1024 × 1024",
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
        # Check if download is complete enough
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

    # ── Seed ──────────────────────────────────────────────────────────────────
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

    # ── Prompt engineering for face/body preservation ─────────────────────────
    final_prompt = req.prompt
    final_negative = req.negative_prompt if req.negative_prompt else FACE_NEGATIVE

    if req.face_preserve or is_img2img:
        # Inject quality tokens that help preserve real face/body features
        face_positive_prefix = (
            "RAW photo, photorealistic, 8k uhd, dslr, soft lighting, "
            "high quality, film grain, Fujifilm XT3, "
        )
        if not any(tok in final_prompt.lower() for tok in ["raw photo", "photorealistic", "8k"]):
            final_prompt = face_positive_prefix + final_prompt
        # Always use face-optimised negative for img2img
        if req.negative_prompt is None or req.negative_prompt.strip() == "":
            final_negative = FACE_NEGATIVE

    try:
        output_type = "latent" if (req.disable_vae and model_choice == "sd15") else "pil"

        # ── Resolution ────────────────────────────────────────────────────────
        target_w = req.width or 512
        target_h = req.height or 512

        if is_img2img:
            raw_b64 = req.image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_data = base64.b64decode(raw_b64)
            init_img = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Resize to model target while keeping aspect-ratio via smart crop
            init_img = ImageOps.fit(
                init_img, (target_w, target_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )

            # Face-preserve mode: slightly sharpen input to help model see face edges
            if req.face_preserve:
                init_img = init_img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=3))

            # Clamp strength: if face_preserve, keep low (0.30–0.55) regardless of user value
            strength = req.strength if req.strength is not None else 0.50
            if req.face_preserve:
                strength = max(0.30, min(strength, 0.60))

            output = img2img_pipe(
                prompt=final_prompt,
                negative_prompt=final_negative,
                image=init_img,
                strength=strength,
                num_inference_steps=max(req.steps, 30),    # minimum 30 for quality
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
