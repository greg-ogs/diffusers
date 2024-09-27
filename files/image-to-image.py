from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch


pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

init_image = load_image('frierern.jpg').resize((512, 512))

prompt = "realistic portrait"
for i in range(10):
    image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    image.save("stable_frieren_" + str(i) + ".png")
