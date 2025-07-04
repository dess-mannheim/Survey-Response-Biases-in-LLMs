# Human Survey Response and Prompt Perturbation Biases in Large Language Models
## Investigating Response and Prompt Perturbation Biases in LLMs: A Survey Design Perspective

## Description
This repository is part of my master thesis project. The project explores human-like response biases and response robustness of Large Language Models in close-ended survey contexts.
Thus, this repository contains all relevant data, perturbation/interview/extraction scripts as well as the three presentations and the final thesis.

The folder structure of this repository is the following:

- `World Value Survey` contains the relevant PDFs of WVS wave 7 questionnaire and codebook

- `assets` contain all bias and non-bias perturbation dataframes, the test (proposal) and final interviews for each model and multiple dataframes after the response extraction. Further, the final questionnaires with the selected 62 questions for the interviews and previous versions of the questionnaire are contained.

**Due to the larger file size, the final and processed interviews with all models can be downloaded here:** -> https://drive.google.com/drive/folders/1isPxAkryGSFvPCBlXXBK469e8KXk08aD?usp=sharing

- `scripts` contains all scripts that I used for:

 1. a-b: Questionnaire Preprocessing and Validation
 2. Test Interviews
 3. a-k: Interviews with each LLM
 4. a-b: Response Extraction with LLM and Regular Expression
 5. Extraction Validation
 6. a-c: Analysis (Final Analysis, Proposal and Midterm, Explanatory Plots, Descriptives)
 