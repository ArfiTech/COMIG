import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline

class ImageGenerationPipeline:
    def __init__(self, model_id_or_path="../anything-v5", device="cuda", torch_dtype=torch.float16):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch_dtype, safety_checker = None)
        self.pipe = self.pipe.to(device)

    def generate_image(self, prompt, input_image_path, output_image_path, strength=0.4, guidance_scale=7.5, image_size=(512, 512)):
        init_image = Image.open(input_image_path).convert("RGB")
        init_image = init_image.resize(image_size)

        images = self.pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
        images[0].save(output_image_path)

if __name__ == "__main__":
    prompt = "1 girl, brown hair, loli, in airplane, laughing, back hair pin"
    input_image_path = "../inputs/arisu_airplane.jpg"
    output_image_path = "./test_results/arisu_laughing.png"

    image_pipeline = ImageGenerationPipeline()
    image_pipeline.generate_image(prompt, input_image_path, output_image_path)