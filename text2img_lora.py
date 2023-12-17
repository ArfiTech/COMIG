from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch
import time

class ImageGenerationPipeline:
    def __init__(self, pretrained_model_name_or_path="../anything-v5", weight_name="0Style-Anime_Screencap.safetensors"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            f"{pretrained_model_name_or_path}", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("../lora_safetensors", weight_name=weight_name)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def generate_image(self, prompt, guidance_scale=7.5, image_size=(512, 512)):
        print("guidance: ", guidance_scale)
        image = self.pipeline(prompt=prompt,
                              num_inference_steps=100,
                              guidance_scale=guidance_scale,
                              width=image_size[0],
                              height=image_size[1]).images[0]
        return image

if __name__ == "__main__":
    # Anime_Screencap
    # Yaro_Artstyle
    prompt="1 girl, school, cool, black hair"
    processor = ImageGenerationPipeline()
    generated_image = processor.generate_image(prompt)

    now = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", now)
    generated_image.save(f"./test_results/t2i_{formatted_time}.png")