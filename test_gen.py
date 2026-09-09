import os
import time
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

print("Importing diffusers...")
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image

model_path = "/home/user/Desktop/stable-diffusion/models/Realistic_Vision_V5.1.safetensors"
print(f"Loading {model_path} with CUDA fp16...")

pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
    local_files_only=True
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.to("cuda")

prompt = "A high-end sports car on a wet neon city street, hyperrealistic, sharp focus"
print(f"Generating test image (VAE DISABLED)... Prompt: {prompt}")

generator = torch.Generator(device="cuda").manual_seed(42)
output = pipe(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=7.0,
    generator=generator,
    output_type="latent"
)

latents = output.images.detach().cpu().to(torch.float32)[0].permute(1, 2, 0)
factors = torch.tensor([
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473]
], dtype=torch.float32)

rgb = torch.matmul(latents, factors)
rgb_min, rgb_max = rgb.min(), rgb.max()
rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
rgb_np = (rgb.numpy() * 255.0).astype(np.uint8)
img = Image.fromarray(rgb_np).resize((512, 512), Image.Resampling.LANCZOS)

out_path = "/home/user/Desktop/stable-diffusion/outputs/test_novae.png"
img.save(out_path)
print(f"SUCCESS! Saved test image to {out_path}")
