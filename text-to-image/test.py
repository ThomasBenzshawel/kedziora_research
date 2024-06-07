from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("./sd-airplane-model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
images = pipeline("model airplane", num_inference_steps=50, guidance_scale=7.5).images

# Save each image individually
for i, image in enumerate(images):
    image.save(f"out_{i}.png")