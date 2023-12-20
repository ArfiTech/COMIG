from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

class ImageGenerationPipeline:
    def __init__(self, model_id="../anything-v5", device="cuda"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker = None)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
    
    def generate_image(self, prompt, num_inference_steps=25, guidance_scale=7, width=512, height=512):
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]
        return image

if __name__ == "__main__":
    prompt = "1girl, outline, draw, chibi, white glasses, blindfolded of glasses, youth, hoodie, laptop, coding, simple background"

    image_pipeline = ImageGenerationPipeline()
    generated_image = image_pipeline.generate_image(prompt)

    generated_image.save("t2i_test.png")