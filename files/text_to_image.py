from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "create a realistic portrait of a girl using a septum, she is wearing a white corset. She has brown skin and short black hair. she doesn't have freckles on her face"

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.show()
image.save("stable_latina_2.png")