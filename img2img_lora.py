from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch
import datetime
import os

class ImageGenerationPipeline:
    def __init__(self, pretrained_model_name_or_path="../anything-v5", weight_name="0Style-Anime_Screencap.safetensors"):
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            f"{pretrained_model_name_or_path}", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("../lora_safetensors", weight_name=weight_name)
        self.weight_name = weight_name

    def generate_image(self, prompts, input_image_path, output_image_path, strength=0.3, guidance_scale=7.5, image_size=(512, 512)):
        init_image = Image.open(input_image_path).convert("RGB")
        init_image = init_image.resize(image_size)
        print("strength, guidance: ", strength, guidance_scale)
        negative_prompt = "worst quality, error"
        
        image = self.pipeline(prompt=prompts,
                              num_inference_steps=100,
                              image=init_image,
                              strength=strength,
                              guidance_scale=guidance_scale,
                              negative_prompt=negative_prompt).images[0]

        image.save(output_image_path)

if __name__ == "__main__":

    init_dir = os.getcwd()

    os.chdir('../inputs')
    img_names = os.listdir()
    imgs = []
    for img in img_names:
        imgs.append(os.path.splitext(img)[0])
    print(imgs) # ['kyouka', 'mahiro', 'kuroneko', 'arisu_airplane', 'arisu_table']
    selected_img = imgs[2]

    os.chdir(init_dir) 

    prompts = "1 girl, loli, cute, crying"
    prompts_list = prompts.split(", ")
    print(prompts_list)

    now = datetime.datetime.today()
    output_path = f"./test_results/{selected_img}_{prompts_list[-1]}_{now.isoformat()}.png"

    init_dir = os.getcwd()
    processor = ImageGenerationPipeline()
    processor.generate_image(prompts=prompts,
                             input_image_path=f"../inputs/{selected_img}.jpg",
                             output_image_path=output_path,
                             image_size=(512, 512))


# from diffusers import AutoPipelineForImage2Image
# from PIL import Image
# import torch
# import datetime
# import os

# init_dir = os.getcwd()

# os.chdir('../inputs')
# img_names = os.listdir()
# imgs = []
# for img in img_names:
#     imgs.append(os.path.splitext(img)[0])
# print(imgs)
# selected_img = imgs[1]

# os.chdir(init_dir)

# prompts = "1 girl, loli, cute, disgust"
# prompt_list = prompts.split(",")
# print(prompt_list)

# pretrained_model_name_or_path = "anything-v5"
# #weight_name = "0Style-Yaro_Artstyle.safetensors"
# weight_name = "0Style-Anime_Screencap.safetensors"

# weight_dict = {"0Style-Yaro_Artstyle.safetensors":"yaro", "0Style-Anime_Screencap.safetensors":"as"}

# input_image_path = f"../inputs/{selected_img}.jpg"
# image_size=(512, 512)

# init_image = Image.open(input_image_path).convert("RGB")
# init_image = init_image.resize(image_size)

# pipeline = AutoPipelineForImage2Image.from_pretrained(f"../{pretrained_model_name_or_path}", torch_dtype=torch.float16, safety_checker = None).to("cuda")
# pipeline.load_lora_weights("../lora_safetensors", weight_name=weight_name)
# image = pipeline(prompt=prompts,
#                  image=init_image,
#                  strength=0.3,
#                  guidance_scale=7.5).images[0]


# now = datetime.datetime.today()
# image.save(f"./test_results/{selected_img}_{prompt_list[-1]}_{weight_dict[weight_name]}_{now.isoformat()}.png")
