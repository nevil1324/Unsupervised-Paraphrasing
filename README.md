# Introduction to NLP 
# Project : Neural Unsupervised Paraphrasing

## Problem Statement :
- Paraphrasing is expressing a sentence using different words while maintaining the meaning. In this project teams will be implementing unsupervised approaches to generate paraphrases for Indian Languages.

## Execution (How to run)
```
1. Statastical Method :
- cd Statastical_Method
-  Run given ipynb file

2. For English Peraphrasing :
- cd Fintuning_For_English_LN
- python3 preprocess.py
- pyhton3 data_loader.py
- main.py

3. For Hindi Peraphrasing :Fintuning_For_Hindi_LN
- cd Fintuning_For_Hindi_LN
- python3 preprocess.py
- pyhton3 data_loader.py
- main.py
```

## Problem Description:
- Paraphrasing, the task of expressing the same meaning using different words or structures, is fundamental in natural language processing (NLP). Paraphrasing can improve the readability and flow of written text. It allows writers to rephrase complex or convoluted sentences into simpler, more concise language, making the text more accessible to a wider audience. Traditional methods for paraphrase generation often rely on supervised approaches, which necessitate large amounts of annotated data. However, acquiring such labeled datasets is labor-intensive and may not cover the diverse range of expressions found in natural language. In contrast, unsupervised paraphrasing aims to generate paraphrases without relying on labeled data, offering a more scalable and versatile solution.

## Dataset : 
- Quora Question Pairs **(QQP)** Dataset (For **English** Text)
- **IndicCorp** (For **Hindi** Text)

## Approach and Implementation:
### (1). Statistical Approach : 
- POS Tagging and Synonym Replacement
- Synonym Retrieval
- Combination of Sentences
- Random Sampling
- Paraphrasing
- Similarity Evaluation

### (2). Using Pre-Trained Language Models
- We take the unsupervised paraphrasing task as a sentence reconstruction task from corrupted input.
- From a sentence, we omit all the stop words to form a corrupted sentence, let's call it Source S, and the original sentence is used as Target T. We use GPT-2 to generate the Target sentence given Source. i.e P(T |S)
- Data Preprocessing :
    - Remove the stop words from the training set.
    - Remove all special characters like, “,”,’
- Sentence Corruption Techniques
    - Random Shuffle
    - Synonym Replacement 
    - Random word Deletion
- Calculates the contextual similarity between two sentences using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model.

## Conclusion
- The statistical method is deemed less efficient compared to the fine-tuning approach applied to pre-trained models. In the statistical method, the focus remains primarily on substituting the most probable words with their synonyms without considering sentence restructuring.
- Conversely, fine-tuning leverages pre-trained models already trained on extensive corpora. It employs a sentence reconstruction task to generate paraphrases by corrupting sentences and attempting to reconstruct the original sentence from the corrupted version.
- However, due to resource constraints, we could only fine-tune the GPT2 pre-trained model for three epochs. Considering the substantial size of GPT2, we faced limitations in fine-tuning it with a larger corpus for more training epochs due to computational issues, leading to less accurate results.
- Additionally, English pre-trained models have fewer parameters compared to their Hindi counterparts, thus performing better in certain contexts.


## Contributors : 
```
1. Nevil Sakhreliya - 2023201005
2. Darshak Devani - 2023201007 
3. Shah Viraj Utpalbhai - 2023201011

Guided By : 
- Prof. Manish Shrivastava 
- Prof. Rahul Mishra
- Teaching Assistant: Lakshmipathi Balaji 
```