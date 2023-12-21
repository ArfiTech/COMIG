from text2text import StoryGenerator, KeywordExtractor
from text2img_lora import ImageGenerationPipeline as TextToImgPipeline
from img2img_lora import ImageGenerationPipeline as ImgToImgPipeline
from scoring import ScoreCont
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import time
import gc
import numpy as np
import torch

from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_np = np.asarray(images)
    images_int = (images_np * 255).astype("uint8")
    images_tensor = torch.from_numpy(images_int).unsqueeze(0)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    # clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    clip_score = clip_score_fn(images_tensor, prompts).detach()
    return round(float(clip_score), 4)

def generate_comic_strips(story_input):
    # Use text2text.py to generate a list of sentences and keywords from the story input
    keyword_extractor = KeywordExtractor()
    scoring = ScoreCont()

    keywords = []
    previous_keywords = []
    sentences = story_input
    for i, sentence in enumerate(sentences):
        prior_words = ['grayscale', 'manga', 'best_quality', 'detailed_face']
        main_words = keyword_extractor.extract_keywords(sentence)
        print("main_words: ", main_words)
        main_words = [item[0] for item in main_words]
        if i == 0:
            previous_keywords.extend(main_words)
        if i > 0 and len(previous_keywords) > 0:
            rm_words = []
            for pre in previous_keywords:
                s = scoring.score_cur_prompt_next_prompt_hdn(pre, sentence)
                print("word, word-sentence sim: ", pre, s)
                if s < 0.3:
                    rm_words.append(pre)
            previous_keywords = [x for x in previous_keywords if x not in rm_words]
        main_words = main_words + previous_keywords
        prior_words.extend(main_words)
        print(f"Original Sentence: {sentence}")
        print(f"Main Words: {prior_words}")
        print("-" * 50)
        keywords.append(prior_words)
    print("keywords: ", keywords)

    # Initialize the text-to-image pipeline
    text_to_img_pipeline = TextToImgPipeline()

    # Initialize the image-to-image pipeline
    img_to_img_pipeline = ImgToImgPipeline()

    # Generate and arrange the comic strips horizontally
    composite_image = Image.new("RGB", (512 * 4 + 30 * 3, 512 + 30), "white")
    draw = ImageDraw.Draw(composite_image)
    font = ImageFont.load_default()

    one_story_score = []
    #print(sentences)
    for i in range(4):
        if i == 0:
            # Generate the first comic strip using text2img.py
            first_keywords = ', '.join(keywords[0])
            first_strip = text_to_img_pipeline.generate_image(first_keywords)
            first_strip.save(f"./result_comics/comic_strip_{i}.png")
            clip_score = calculate_clip_score(first_strip, sentences[i])
            print(f"comic_strip_{i} & {sentences[i]}: ", clip_score)
            one_story_score.append(clip_score)
        else:
            score = scoring.score_cur_prompt_next_prompt_hdn(sentences[i-1], sentences[i])
            print(f"{i}th score: ", score)
            if score >= 0.9:
                prior_image_path = f"./result_comics/comic_strip_{i-1}.png"
                output_image_path = f"./result_comics/comic_strip_{i}.png"
                next_strip_keywords = ', '.join(keywords[i])
                strength = (1 - score)*2.5
                strip = img_to_img_pipeline.generate_image(next_strip_keywords, prior_image_path, output_image_path, strength=strength, guidance_scale=12.5)
                strip.save(f"./result_comics/comic_strip_{i}.png")
            else:
                score_list = []
                for j in range(i):
                    if i == j:
                        break
                    score = scoring.score_cur_prompt_next_prompt_hdn(sentences[j], sentences[i])
                    score_list.append(score)
                    print(f"score{j}{i}: {score}")
                if max(score_list) >= 0.6:
                    print(f"{i}th")
                    prior_image_path = f"./result_comics/comic_strip_{score_list.index(max(score_list))}.png"
                    output_image_path = f"./result_comics/comic_strip_{i}.png"
                    next_strip_keywords = ', '.join(keywords[i])
                    #strength = (1-max(score_list))*2.5
                    strength = -(np.exp(-14/(max(score_list)*10))) + 1
                    strip = img_to_img_pipeline.generate_image(next_strip_keywords, prior_image_path, output_image_path, strength=strength, guidance_scale=20.5)
                    strip.save(f"./result_comics/comic_strip_{i}.png")
                else:
                    print(f"else {i}th")
                    prior_image_path = f"./result_comics/comic_strip_{score_list.index(max(score_list))}.png"
                    output_image_path = f"./result_comics/comic_strip_{i}.png"
                    next_strip_keywords = ', '.join(keywords[i])
                    strength = -(np.exp(-(1-max(score_list))*10)) + 1
                    strip = img_to_img_pipeline.generate_image(next_strip_keywords, prior_image_path, output_image_path, strength=strength, guidance_scale=7.5)
                    strip.save(f"./result_comics/comic_strip_{i}.png")
                clip_score = calculate_clip_score(strip, sentences[i])
                print(f"comic_strip_{i} & {sentences[i]}: ", clip_score)
                one_story_score.append(clip_score)

        # Paste the comic strip onto the composite image
        strip_image = Image.open(f"./result_comics/comic_strip_{i}.png")
        composite_image.paste(strip_image, (i * (512 + 30), 0))

    now = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", now)
    composite_image.save(f"./result_comics/composite_image_{formatted_time}.png")
    # print("One manga score: ", one_story_score)
    print("Composite image generated successfully!")
    return one_story_score

if __name__ == "__main__":
    csv_story = pd.read_csv('./input_prompt_cut.csv')
    stories = csv_story[['story1', 'story2', 'story3', 'story4']]
    print("len: ", len(stories))
    all_mangas_score = []
    for j in range(len(stories)):
        story = stories.loc[[j]]
        stories_4line = []
        for i in story:
            stories_4line.append(story[i][j])
        print(f"{j}th stories: ", stories_4line)
        manga_score = generate_comic_strips(stories_4line)
        print(f"{j}th stories score: ", manga_score)
        all_mangas_score.append(sum(manga_score)/4)
    print("each manga's score: ", all_mangas_score)
    print("all clip score: ", sum(all_mangas_score)/len(stories))

