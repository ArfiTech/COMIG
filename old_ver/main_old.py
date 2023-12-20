from text2text import StoryGenerator, KeywordExtractor
from text2img import ImageGenerationPipeline as TextToImgPipeline
from img2img import ImageGenerationPipeline as ImgToImgPipeline
from PIL import Image, ImageDraw, ImageFont
import time

def generate_comic_strips(story_input):
    # Use text2text.py to generate a list of sentences and keywords from the story input
    story_generator = StoryGenerator()
    keyword_extractor = KeywordExtractor()

    # sentences = story_generator.generate_story(story_input)
    sentences = story_generator.break_down_long_sentence(story_input)
    keywords = []
    print("Generated Stories:")
    for sentence in sentences:
        main_words = keyword_extractor.extract_keywords(sentence)
        main_words = [item[0] for item in main_words]
        print(f"Original Sentence: {sentence}")
        print(f"Main Words: {main_words}")
        print("-" * 50)
        keywords.append(main_words)
    print("keywords: ", keywords)

    # Initialize the text-to-image pipeline
    text_to_img_pipeline = TextToImgPipeline()

    # Initialize the image-to-image pipeline
    img_to_img_pipeline = ImgToImgPipeline()

    # Generate and arrange the comic strips horizontally
    composite_image = Image.new("RGB", (512 * 4 + 30 * 3, 512 + 30), "white")
    draw = ImageDraw.Draw(composite_image)
    font = ImageFont.load_default()

    for i in range(4):
        if i == 0:
            # Generate the first comic strip using text2img.py
            first_keywords = ', '.join(keywords[0])
            first_strip = text_to_img_pipeline.generate_image(first_keywords)
            first_strip.save(f"./result_comics/comic_strip_{i}.png")
        else:
            # Generate the next three comic strips using img2img.py
            prior_image_path = f"./result_comics/comic_strip_{i-1}.png"
            output_image_path = f"./result_comics/comic_strip_{i}.png"
            next_strip_keywords = ', '.join(keywords[i])
            img_to_img_pipeline.generate_image(next_strip_keywords, prior_image_path, output_image_path)

        # Paste the comic strip onto the composite image
        strip_image = Image.open(f"./result_comics/comic_strip_{i}.png")
        composite_image.paste(strip_image, (i * (512 + 30), 0))

        # Draw the Sentence below each comic strip
        text_width, text_height = draw.textsize(f"\"{sentences[i]}\"", font=font)
        x_position = i * (512 + 30) + (512 - text_width) // 2
        y_position = 512 + 10  # Adjusted for vertical centering
        draw.text((x_position, y_position), f"Scene: {sentences[i]}", fill="black", font=font)

    now = time
    composite_image.save(f"./result_comics/composite_image_{now.localtime()}.png")
    print("Composite image generated successfully!")

if __name__ == "__main__":
    #user_story_input = "In a charming little town nestled between rolling hills and blooming meadows, there's a cozy bakery named \"Sweet Whiskers.\" The bakery is known for its delightful pastries and warm atmosphere. One sunny morning, a curious kitten named Mochi, with soft fur and sparkling eyes, stumbles upon the bakery's open door. Intrigued by the sweet aroma of freshly baked goods, Mochi decides to explore the inviting space. The friendly bakers, noticing the adorable visitor, offer Mochi a tiny pastry topped with a sprinkle of love. Overwhelmed by the cuteness of the moment, the customers can't help but smile as Mochi enjoys the treat, creating a heartwarming scene that brightens everyone's day."
    user_story_input = "A batter enters the batter's box and waits for the pitcher to throw the ball. The pitcher throws the ball and the batter swings the bat. The batter's bat strikes the ball and the ball flies out for a home run. The scoreboard shows the team's score going up."
    generate_comic_strips(user_story_input)
