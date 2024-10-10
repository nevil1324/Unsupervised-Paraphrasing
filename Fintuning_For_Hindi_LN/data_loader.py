from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, random_split


class DataAugmentationTechnique:
  ########################################################################
  # Randomly swap the words from sentence
  ########################################################################
    def random_word_swap(self,sentence):
        random.shuffle(sentence)
        return sentence
    
    def random_deletion(self,words, p):
        if len(words) == 0:
            return words
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        return new_words
    

class HindiPeraphraseDataset:
    def __init__(self, sentences,tokenizer, max_length=100):
        self.sentences = sentences
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.IGNORE_INDEX = -100
        self.tokenizer = tokenizer
        self.dat_obj = DataAugmentationTechnique()
        for sentence in sentences:
            sentence_to_corrupt = deepcopy(sentence)
            sentence_to_delete = deepcopy(sentence)
            for i in range(2):
                corrupted = self.dat_obj.random_word_swap(sentence_to_corrupt)
    #             print(sentence)
    #             print(corrupted)
    #             print()
                sentence_ignore_len = len(self.tokenizer('[BOS]' + " ".join(corrupted) + "[SEP]")["input_ids"])
                encodings = self.tokenizer('[BOS]' + " ".join(corrupted) + "[SEP]" + " ".join(sentence) + '[EOS]', truncation=True,
                                            max_length=max_length, padding="max_length", add_special_tokens=True)
                input_ids = torch.tensor(encodings['input_ids'])
                attention_mask = torch.tensor(encodings['attention_mask'])
                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)

                label = deepcopy(input_ids)
                label[:sentence_ignore_len] = self.IGNORE_INDEX
                self.labels.append(label)
            for i in range(1):
                corrupted = self.dat_obj.random_deletion(sentence_to_delete,0.3)
                if len(corrupted) > 0:
                    sentence_ignore_len = len(self.tokenizer('[BOS]' + " ".join(corrupted) + "[SEP]")["input_ids"])
                    encodings = self.tokenizer('[BOS]' + " ".join(corrupted) + "[SEP]" + " ".join(sentence) + '[EOS]', truncation=True,
                                                max_length=max_length, padding="max_length", add_special_tokens=True)
                    input_ids = torch.tensor(encodings['input_ids'])
                    attention_mask = torch.tensor(encodings['attention_mask'])
                    self.input_ids.append(input_ids)
                    self.attention_mask.append(attention_mask)

                    label = deepcopy(input_ids)
                    label[:sentence_ignore_len] = self.IGNORE_INDEX
                    self.labels.append(label)


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'attention_mask': self.attention_mask[idx]  # unsqueeze to add batch dimension
        }
    