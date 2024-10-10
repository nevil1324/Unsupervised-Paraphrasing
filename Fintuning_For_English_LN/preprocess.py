import csv
import os
import random
import argparse
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('stopwords')

english_stopwords = stopwords.words('english')

# Stopwords from the paper
# 1. From case study (that should not be replaced)
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', '']

english_stopwords += ['someone', 'something', 'make', 'see', 'everything', 'anyone', 'anything', 'everyone']

tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()

def remove_stopwords(sentence):
    sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence
                if word.lower() not in english_stopwords]
    sentence = ' '.join(sentence)
    sentence = sentence.replace("''", '"').replace('``', '"')
    sentence = detokenizer.detokenize(sentence.split())
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for le in syn.lemmas():
            synonym = le.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in 'abcdefghijklmnopqrstuvwxyz'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    new_words = words.copy()
    random_words_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_words_list)
    num_replaced = 0
    for random_word in random_words_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: # only n words replaced
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return words

def sentence_corrupting(sentence, shuffle_ratio=0.5, replace_ratio=0.4):
    # Synonym Replacement
    words = sentence.split()
    no_syn = max(1, int(len(words)*replace_ratio))
    words = synonym_replacement(words, no_syn)

    # Random Shuffling
    if random.random() < shuffle_ratio:
        random.shuffle(words)

    return ' '.join(words)

def prepare_data(input_file, output_file, corrupted_output=None, max_length=1024, seed=1234, limit=2000):
    random.seed(seed)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    data = []

    df = pd.read_csv(input_file)

    # Extract the "question1" col
    questions = df['question1'].tolist()
    
    for que in tqdm(questions[:limit], desc="preprocessing..."):
        sentence = que.strip()
        corrupted_sentence = remove_stopwords(sentence)
        total_line = corrupted_sentence + '\n' + sentence
        if len(gpt_tokenizer.encode(total_line)) < max_length:
            data.append([corrupted_sentence, sentence])
        if len(data) >= limit:
            break 
    
    # Write [original, corrupted] to  csv
    with open(corrupted_output, 'w') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in tqdm(data, total = limit, desc = "writing corrupting sentences..."):
            # Corrupting Function call
            corrupted = sentence_corrupting(corrupted)
            writer.writerow([corrupted, sentence])


##########################
# corrupting Training data
##########################

train_input_file = 'train.csv.zip'
corrupted_output_file = 'train_preprocessed_corrupted.csv'
max_length = 1024
seed = 1234
limit = 10000
prepare_data(train_input_file, corrupted_output_file, max_length, seed, limit = limit)

##########################
# corrupting Testing data
##########################

test_input_file = 'test.csv.zip'
test_corrupted_output_file = 'test_preprocessed_corrupted.csv'
max_length = 1024
seed = 1234
limit = 1000
prepare_data(test_input_file, test_corrupted_output_file, max_length, seed, limit = limit)