import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from keybert import KeyBERT

class StoryGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)

    def generate_story(self, input_sentence, num_stories=3):
        all_stories = []
        all_stories.append(input_sentence)
        for _ in range(num_stories):
            input_ids = self.tokenizer.encode(all_stories[-1], return_tensors='pt')
            beam_output = self.model.generate(
                input_ids, 
                max_length=1000, 
                num_beams=20,
                no_repeat_ngram_size=8,
                early_stopping=True
            )
            generated_story = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
            generated_story = generated_story.split('.')[1]
            generated_story = generated_story.replace('\n', '')
            generated_story = generated_story.replace('"', '')
            all_stories.append(generated_story)
        return all_stories
    
    def break_down_long_sentence(self, long_sentence):
        input_ids = self.tokenizer.encode(long_sentence, return_tensors='pt')
        beam_output = self.model.generate(
            input_ids, 
            max_length=1000, 
            num_beams=20,
            no_repeat_ngram_size=8,
            early_stopping=True
        )
        generated_story = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        short_sentences = generated_story.split('.')[:4]
        short_sentences = [s.strip() for s in short_sentences if s.strip()]
        return short_sentences

class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()

    def extract_keywords(self, sentence):
        main_words = self.kw_model.extract_keywords(sentence)
        return main_words

if __name__ == "__main__":
    # prompt = "A cute girl is eating a bread."
    prompt = "In a charming little town nestled between rolling hills and blooming meadows, there's a cozy bakery named \"Sweet Whiskers.\" The bakery is known for its delightful pastries and warm atmosphere. One sunny morning, a curious kitten named Mochi, with soft fur and sparkling eyes, stumbles upon the bakery's open door. Intrigued by the sweet aroma of freshly baked goods, Mochi decides to explore the inviting space. The friendly bakers, noticing the adorable visitor, offer Mochi a tiny pastry topped with a sprinkle of love. Overwhelmed by the cuteness of the moment, the customers can't help but smile as Mochi enjoys the treat, creating a heartwarming scene that brightens everyone's day."

    story_generator = StoryGenerator()
    # generated_stories = story_generator.generate_story(prompt)
    short_sentences = story_generator.break_down_long_sentence(prompt)
    print(short_sentences)

    keyword_extractor = KeywordExtractor()

    print("Generated Stories:")
    for sentence in short_sentences:
        main_words = keyword_extractor.extract_keywords(sentence)
        main_words = [item[0] for item in main_words]
        print(f"Original Sentence: {sentence}")
        print(f"Main Words: {main_words}")
        print("-" * 50)