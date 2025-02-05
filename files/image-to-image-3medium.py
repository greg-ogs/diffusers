import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import time

token = input("Enter your token: ")
login(token)

time01 = time.time()
location = "/app/stable_diffusion_3.5_medium_diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
    cache_dir="/app/cache",
)
pipe.enable_model_cpu_offload()
time02 = time.time()
load_time = time02 - time01
print("Load time: " + str(load_time))

for i in range(1, 6):
    time03 = time.time()
    image = pipe(
        prompt="A gothic woman with long, flowing black hair and pale skin. The image has a dark fantasy, inked comic book style with high contrast and sharp details. Cool-toned gradient.",
        negative_prompt="",
        num_inference_steps=40,
        guidance_scale=7.5,
    ).images[0]

    image.save(f"image_{i}.png")
    time04 = time.time()
    inference_time = time04 - time03
    print(f"Inference time for image {i}: " + str(inference_time))