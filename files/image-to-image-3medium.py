import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
import time
from diffusers.utils import load_image
#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
token = input("Enter your token: ")
login(token)

time01 = time.time()
location = "/opt/project/stable_diffusion_3_medium_diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, cache_dir="/opt/project/cache")  #
pipe.enable_model_cpu_offload()
time02 = time.time()
load_time = time02 - time01
print("Load time: " + str(load_time))

time03 = time.time()
image = pipe(
    prompt="a realistic portrait photo of a skinny ginger female model, white skin and freckles, she is smiling and is wearing a red dress with an open back and low neckline.",
    # "A photo of batman using his batarang to fight a gotzilla attacking tokio"
    # "a portrait photo of a ginger female model wearing a gray sweater and glasses"
    # "A portrait of a cute girl in a League of legends jinx cosplay"
    negative_prompt="",
    num_inference_steps=37,
    height=800,
    width=800,
    guidance_scale=7.0,
).images[0]

image.save("test_image.png")
time04 = time.time()
inference_time = time04 - time03
print("Inference time is: " + str(inference_time))
print("Total time taken: " + str(inference_time + load_time))


# init_image = load_image('samara.png').resize((512, 512))
#
# prompt = "Portrait as 90s anime aesthetic"
# image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5,
#              height=1200, width=1200, guidance_scale=0.0).images[0]
# image.save("sd3_hello_world_5.png")