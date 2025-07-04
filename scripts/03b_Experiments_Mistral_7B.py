import torch
# Select local hardware device to use
if torch.cuda.is_available():
    device = torch.device("cuda:0")      # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")         # Apple silicon GPU
else:
    device = torch.device("cpu")         # CPU


# this is the script executing the experimental interviews

import pandas as pd
pd.set_option('display.max_colwidth', None)

main_question = ["question"]
main_answer = ["list_answer_options"]

bias_answer_columns = ["list_answer_no_refusal","list_answer_no_middle","list_answer_options_reversed", "list_answer_options", "original_personality_prime", "original_emotional_prime"]
non_bias_question_columns = ["question_typo", "question_letter_swap","question_synonym","question_paraphrased", "question_keyboardtypo"]

# read the preprocessed questionnaire
questionnaire =  pd.read_excel('../assets/20240924_selected_questionnaire.xlsx', engine='openpyxl')
# subset the questionnaire with the ten selected questions
questionnaire = questionnaire[questionnaire["selected"]==1]

torch.cuda.empty_cache()

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", device=0)

# create empty response list which is transformed to a dataframe
response_list = []

# define rounds to be tested
rounds = 25

for round in range(0,rounds):
  print("Current round: ", round)
  torch.cuda.empty_cache()
  for row in range(0,len(questionnaire)):
    question_ID = questionnaire.iloc[row]["full_question_ID"]
    for i,k in enumerate(bias_answer_columns):
        # first retrieve original question
        question = questionnaire.iloc[row]["question"]
        # second retrieve bias perturbed answer options from the questionnaire
        answer_options = questionnaire.iloc[row][k]
        messages = [{"role": "user",
                        "content": f"""Answer the following question: {question} 
                        This is a list of possible answer options: {answer_options}
                        You must pick one of the answer options. Only answer with the label.
                        """},
                    ]
        answer = pipe(messages, max_length=1000, max_new_tokens=10)
        response = answer[0]["generated_text"][1]["content"]
        print(response)
        df_response_input = [round,question_ID, question, answer_options, k,  response]
        response_list.append(df_response_input)
        torch.cuda.empty_cache()
    for p,n in enumerate(non_bias_question_columns):
            # first retrieve non-bias perturbed question formats
            question = questionnaire.iloc[row][n]
            # second retrieve original answer option scale
            answer_options = questionnaire.iloc[row]["list_answer_options"]
            messages = [{"role": "user",
                        "content": f"""Answer the following question: {question} 
                        This is a list of possible answer options: {answer_options}
                        You must pick one of the answer options. Only answer with the label.
                        """},
                        ]
            answer = pipe(messages, max_length=1000, max_new_tokens=10)
            response = answer[0]["generated_text"][1]["content"]
            print(response)
            df_response_input = [round,question_ID, question, answer_options, n, response]
            response_list.append(df_response_input)
            torch.cuda.empty_cache()

# create an empty dataframe which is filled with the model responses to the q&a combinations
response_columns = ["round","question_ID", "question", "answer_options", "type", "response"]
model_response = pd.DataFrame(response_list, columns=response_columns)
model_response["model_name"] = "Mistral-7B-Instruct-v0.3"

from datetime import datetime

def get_date_string():
    # Get the current date
    now = datetime.now()
    
    # Format the date as YYYYMMDD
    date_string = now.strftime('%Y%m%d')
    
    return date_string

current_date = get_date_string()
model_response.to_csv(f"../assets/test_interviews/{current_date}_mistral_7B_interviews.csv", encoding="utf-8", sep=";")