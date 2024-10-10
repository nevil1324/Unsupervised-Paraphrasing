from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, random_split



class LoadDataset:
    def __init__(self):
        self.filename = '/kaggle/input/inlp-proj/new.txt'
        self.hindi_stopwords = {
    "एक", "एवं", "यह", "इस", "के", "का", "की", "को", "में", "है", "हैं", "कर", "किया", "किए", "करते", "करना", "किसी", "गया", "जाता", "जाती", "जाते", "साथ", "अपने", "हुआ", "होता", "होती", "होते", "वाले", "वह", "वहाँ", "जैसा", "जिसका", "जिसकी", "जिसके", "जिनका", "जिनकी", "जिनके", "तथा", "उसके", "उसका", "उसकी", "उनके", "उनका", "उनकी", "उनको", "कुछ", "इसका", "इसकी", "इसके", "सभी", "अगर", "इसमें", "उनका", "उनकी", "उनके", "जैसे", "जिसमें", "तिन्हों", "तिन्हें", "पहले", "बाद", "मानो", "अंदर", "भीतर", "पूरे", "खुद", "आप", "अब", "जब", "जहाँ", "जितना", "जितने", "तब", "वहीं", "हुआ", "होता", "होती", "वाला", "वाली", "वाले"
}
        self.sentences = self.read_sentences_from_file()
        

        
    def read_sentences_from_file(self, max_lines=10, lang="hi", remove_nuktas=False):
        sentences = []
        normalizer_factory = IndicNormalizerFactory()
        normalizer = normalizer_factory.get_normalizer(lang, remove_nuktas=remove_nuktas)

        with open(self.filename, encoding="utf-8") as f:
            for _, line in zip(range(4000), f):
                normalized_line = normalizer.normalize(line.strip())
                tokens = indic_tokenize.trivial_tokenize(normalized_line)
                sentences.append(tokens)        
       
        sentences=self.preprocess(sentences)      
        return sentences
    
    
    def preprocess(self, sentences):
        preprocessed_sentences = []
        for sentence in sentences:
            # Replace specific characters
            sentence = [word.replace("’", "").replace("'", "").replace("-", " ") for word in sentence]
            # Remove stopwords
            sentence = [word for word in sentence if word not in self.hindi_stopwords]
            preprocessed_sentences.append(sentence)
        return preprocessed_sentences

    
    def getdata(self):
        return self.sentences
