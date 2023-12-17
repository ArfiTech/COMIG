import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertTokenizer, BertForSequenceClassification
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
            num_beams=16,
            no_repeat_ngram_size=8,
            early_stopping=True
        )
        generated_story = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        print(generated_story)
        short_sentences = generated_story.split('.')[:4]
        short_sentences = [s.strip() for s in short_sentences if s.strip()]
        return short_sentences

class KeywordExtractor:
    def __init__(self):
        self.kw_model = KeyBERT()

    def extract_keywords(self, sentence):
        main_words = self.kw_model.extract_keywords(sentence)
        return main_words

from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Summarizer:
    def __init__(self):
        self.model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def summarize_story(self, long_story, num_sentence=4):
        input_ids = self.tokenizer.encode("summarize: " + long_story, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(input_ids, max_length=200, num_beams=4, length_penalty=2.0, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # Splitting the summary into 4 sentences
        summary_sentences = summary.split('.')[:4]
        summary_sentences = [s.strip() for s in summary_sentences if s.strip()]

        return summary_sentences

if __name__ == "__main__":
    # prompt = "A cute girl is eating a bread."
    prompt = "A batter enters the batter's box and waits for the pitcher to throw the ball. The pitcher throws the ball and the batter swings the bat. The batter's bat strikes the ball and the ball flies out for a home run. The scoreboard shows the team's score going up."

    summarizer = T5Summarizer()

    # 스토리 요약
    summary = summarizer.summarize_story(prompt)

    # 결과 출력
    print("Original Story:\n", prompt)
    print("\nSummarized Story:\n", summary)
    
    # story_generator = StoryGenerator()
    # # generated_stories = story_generator.generate_story(prompt)
    # short_sentences = story_generator.break_down_long_sentence(prompt)
    # print(short_sentences)

    # keyword_extractor = KeywordExtractor()

    # print("Generated Stories:")
    # for sentence in short_sentences:
    #     main_words = keyword_extractor.extract_keywords(sentence)
    #     main_words = [item[0] for item in main_words]
    #     print(f"Original Sentence: {sentence}")
    #     print(f"Main Words: {main_words}")
    #     print("-" * 50)