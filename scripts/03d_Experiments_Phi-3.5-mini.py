import pandas as pd
pd.set_option('display.max_colwidth', None)

main_question = ["question"]
main_answer = ["list_answer_options"]

bias_answer_columns = ["list_answer_no_refusal","list_answer_no_middle", "list_answer_middle", "list_answer_options_reversed", "list_answer_options", "original_personality_prime", "original_emotional_prime"]
non_bias_question_columns = ["question_typo", "question_letter_swap","question_synonym","question_paraphrased", "question_keyboardtypo"]

# delayed experiments with middle category
#bias_answer_columns = ["list_answer_middle"]
#non_bias_question_columns = []

# read the preprocessed questionnaire
#questionnaire =  pd.read_excel('../assets/20241029_selected_questionnaire.xlsx', engine='openpyxl')
questionnaire =  pd.read_csv('../assets/20241029_selected_questionnaire.csv', sep=";")
# subset the questionnaire with the ten selected questions
questionnaire = questionnaire[questionnaire["selected"]==1]
print(questionnaire.shape)
questionnaire.head(2)

# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct", device=0)
pipe(messages, max_new_tokens=10)

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

# create a bias interaction experiment where there is a question-side and answer option-side bias
# here we use
perturbed_question = "list_answer_options_reversed"
perturbed_answer = "question_paraphrased"

for round in range(0,rounds):
  print("Current round: ", round)
  torch.cuda.empty_cache()
  for row in range(0,len(questionnaire)):
      question_ID = questionnaire.iloc[row]["full_question_ID"]
      # first retrieve original question
      question = questionnaire.iloc[row]["question_paraphrased"]
      # second retrieve bias perturbed answer options from the questionnaire  
      answer_options = questionnaire.iloc[row]["list_answer_options_reversed"]      
      messages = [{"role": "user",
                        "content": f"""Answer the following question: {question} 
                        This is a list of possible answer options: {answer_options}
                        You must pick one of the answer options. Only answer with the label.
                        """},
                      ]    
      answer = pipe(messages, max_length=1000, max_new_tokens=10)       
      response = answer[0]["generated_text"][1]["content"]         
      print(response)  
      df_response_input = [round,question_ID, question, answer_options, "interaction_paraphrase_reversed",  response]
      response_list.append(df_response_input)
      torch.cuda.empty_cache()

# create an empty dataframe which is filled with the model responses to the q&a combinations
response_columns = ["round","question_ID", "question", "answer_options", "type", "response"]
model_response = pd.DataFrame(response_list, columns=response_columns)
model_response["model_name"] = "Phi-3.5-mini"

from datetime import datetime

def get_date_string():
    # Get the current date
    now = datetime.now()
    
    # Format the date as YYYYMMDD
    date_string = now.strftime('%Y%m%d')
    
    return date_string

current_date = get_date_string()
model_response.to_csv(f"../assets/final_interviews/{current_date}_Phi-3.5-mini_selected_interviews.csv", encoding="utf-8", sep=";")