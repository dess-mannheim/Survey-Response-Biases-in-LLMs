# load packages
import os
import pandas as pd
import re
import numpy as np
import seaborn as sns
import scipy as sci
import matplotlib.pyplot as plt

# define the path to the responses from the test interviews
interview_path = "../assets/final_interviews/"

# model names
model_names = ["gemini_1.5", "Llama-3.1", "mistral_7B", "gpt2", 
               "gemma-2", "Gwen2.5", "Phi-3.5", "Yi-1.5"]

# define a function for retrieving the responses of interest

def find_matching_files(directory, string_list):
    matching_files = []
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if any part of the filename is in the list of strings
        if any(substring in filename for substring in string_list):
            matching_files.append(filename)
    
    return matching_files
final_interviews_paths = find_matching_files(interview_path, model_names)
# create an empty list to add all interviews and convert to a dataframe afterwards
interviews_list = []
num_interviews = []

for i in final_interviews_paths:
    df_interview = pd.read_csv(interview_path+i, sep = ";")
    num_interviews.append(len(df_interview))
    interviews_list.append(df_interview)
    print("Successfully appended: ", i)
    
# Concatenate them by rows
final_interviews = pd.concat(interviews_list, ignore_index=True)

# check length/number of interviews conducted
print("Number of interviews per model:\n", final_interviews["model_name"].value_counts())

# get sum of all interviews conducted
print("------------------------------ \nSum of all interviews:", sum(num_interviews))

# Just load once in quantized version
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", 
    quantization_config=quantization_config
)
# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=0)

# create empty response list which is transformed to a dataframe
response_list = []

import numpy as np
final_interviews["mistral_extraction"] = np.nan*len(final_interviews)

for row in range(0,len(final_interviews)):
    response = final_interviews.iloc[row]["response"]
    messages = [{"role": "user", 
                 "content": 
                 f"""Find the first integer, positive or negative, in the text. 
                 Return only this number!
                 Do not include any other words or characters! 
                 If no number is present, output "None":
                 {response}
                 """},
                ]
    answer = pipe(messages, max_length=1000, max_new_tokens=10)
    response = answer[0]["generated_text"][1]["content"]
    print(response)
    final_interviews.iloc[row, final_interviews.columns.get_loc('mistral_extraction')] = response
    response_list.append(response)
    torch.cuda.empty_cache()

# save dataframe to csv
final_interviews.to_csv("../assets/final_interviews/ALL_Llama-3.2-3B-processed_selected_interviews.csv", sep=";")
