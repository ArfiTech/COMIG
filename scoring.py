import torch
import transformers
import math
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class ScoreCont:
    def __init__(self):
        pass
    
    # BertForNextSentencePrediction: 두 문장이 이어져 있는지 여부를 예측하기 위한 모델로, 두 문장이 이어져 있다고 예측하는 값이 1에 가까움
    def score_cur_prompt_next_prompt_tf(self, cur_prompt, next_prompt):
        model = transformers.BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        
        input_text = f"{cur_prompt} [SEP] {next_prompt} [SEP]"
        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
        
        outputs = model(input_ids)
        print("model output: ", outputs)
        logits = outputs[0][0][0].item()
        score = 1 / (1 + math.exp(-logits))
        return score
    
    def score_cur_prompt_next_prompt_hdn(self, cur_prompt, next_prompt):
        # BERT 모델 및 토크나이저 초기화
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        
        # 입력 문장을 토큰화하고 텐서로 변환
        input_ids = torch.tensor([tokenizer.encode(cur_prompt, add_special_tokens=True)])
        answer_ids = torch.tensor([tokenizer.encode(next_prompt, add_special_tokens=True)])
        
        # BERT 모델의 hidden states 추출
        with torch.no_grad():
            outputs = model(input_ids)
            question_hidden_states = outputs[0].squeeze(0)  # 첫 번째 문장의 hidden states
            outputs = model(answer_ids)
            answer_hidden_states = outputs[0].squeeze(0)  # 첫 번째 문장의 hidden states

        # 벡터 값을 추출하여 코사인 유사도 계산
        question_vector = torch.mean(question_hidden_states, dim=0).numpy()
        answer_vector = torch.mean(answer_hidden_states, dim=0).numpy()
        similarity = cosine_similarity([question_vector], [answer_vector])[0][0]

        return similarity


if __name__ == "__main__":
    '''
    question = "What motivated you to apply for our company?"
    answer = "Your company is showing very rapid growth. The current size is small, but I want to work with you by spreading the possibility of future growth."
    '''

    story = [
        "A batter enters the batter's box and waits for the pitcher to throw the ball.",
        "The pitcher throws the ball and the batter swings the bat.",
        "The batter's bat strikes the ball and the ball flies out for a home run.",
        "The scoreboard shows the batter's team's score going up.",
        "A cute high school girl confesses to me.",
    ]

    scoring = ScoreCont()
    n = len(story)
    score_list = []
    for i in range(1,n):
        for j in range(i):
            if i == j:
                break
            score = scoring.score_cur_prompt_next_prompt_hdn(story[j], story[i])
            score_list.append(score)
            print(f"score{j}{i}: {score}")
            
    
    
    score1_12 = scoring.score_cur_prompt_next_prompt_tf(story[0], story[1])
    score1_23 = scoring.score_cur_prompt_next_prompt_tf(story[1], story[2])
    score1_34 = scoring.score_cur_prompt_next_prompt_tf(story[2], story[3])
    score1_45 = scoring.score_cur_prompt_next_prompt_tf(story[3], story[4])
    score2_12 = scoring.score_cur_prompt_next_prompt_hdn(story[0], story[1])
    score2_23 = scoring.score_cur_prompt_next_prompt_hdn(story[1], story[2])
    score2_34 = scoring.score_cur_prompt_next_prompt_hdn(story[2], story[3])
    score2_45 = scoring.score_cur_prompt_next_prompt_hdn(story[3], story[4])
    print("score1: ", score1_12)
    print("score1: ", score1_23)
    print("score1: ", score1_34)
    print("score1: ", score1_45)
    print("score2: ", score2_12)
    print("score2: ", score2_23)
    print("score2: ", score2_34)
    print("score2: ", score2_45)
