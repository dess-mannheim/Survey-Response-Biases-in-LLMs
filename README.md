# Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses

## Description
This repository belongs to the project [**Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses**](https://doi.org/10.48550/arXiv.2507.07188). The project explores human-like response biases and response robustness of Large Language Models in close-ended survey contexts.
This repository contains all relevant data, perturbation/interview/extraction scripts as well as the three presentations and the final thesis.

The folder structure of this repository is the following:

- `World Value Survey` contains the relevant PDFs of WVS wave 7 questionnaire and codebook

- `assets` contain all bias and non-bias perturbation dataframes, the test (proposal) and final interviews for each model and multiple dataframes after the response extraction. Further, the final questionnaires with the selected 62 questions for the interviews and previous versions of the questionnaire are contained.

- `scripts` contains all scripts that I used for:

 1. a-b: Questionnaire Preprocessing and Validation
 2. Test Interviews
 3. a-k: Interviews with each LLM
 4. a-b: Response Extraction with LLM and Regular Expression
 5. Extraction Validation
 6. a-c: Analysis (Final Analysis, Proposal and Midterm, Explanatory Plots, Descriptives)
 
