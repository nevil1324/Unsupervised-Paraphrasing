import math
import os
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
# !pip uninstall transformers
# !pip uninstall accelerate
#!pip install transformers[torch]
import json
import random
import numpy as np
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from data_loader import QQPDataset
import gc
gc.collect()
# torch.cuda.empty_cache()

# Set all seeds
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class FinetuneGPT2(object):
    def __init__(self, args):
        self.args = args
        self.special_tokens_dict = {'sep_token': '[SEP]'}
        self.device = self.args.device
        self.model = self.tokenizer = None
        self.global_step = None

    def build_model(self, checkpoint_dir=None, with_tokenizer=True):
        if checkpoint_dir is None or with_tokenizer is False:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.model)
            self.tokenizer.add_special_tokens(self.special_tokens_dict)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # If resume Training from prev checkpoint
            self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)

        if checkpoint_dir is None:
            self.model = GPT2LMHeadModel.from_pretrained(self.args.model)
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
        
        self.model.to(self.device)
        # Set model Train Mode
        self.model.train()

        self.global_step = 0
        if hasattr(self.args, 'summary_dir'):
            self.writer = SummaryWriter(self.args.summary_dir)

    def generate_text(self, input_texts, max_length=1024, decoding='greedy',
                      suffix=''):
        # set model to Eval mode
        self.model.eval()
        sentences_list = []
        with torch.no_grad():
            kwargs = {'max_length': max_length}
            if decoding == 'sampling':
                kwargs['do_sample'] = True
                kwargs['top_k'] = 0
                kwargs['top_p'] = 1
                kwargs['temperature'] = 1
                kwargs['num_return_sequences'] = 10
            for input_text in input_texts:
                sequences = []
                input_text = input_text + suffix
                input_encoding = self.tokenizer.encode(
                    input_text, return_tensors='pt')
                input_encoding = input_encoding.to(self.device)
                # generating Tokens
                generated_tokens = self.model.generate(
                    input_encoding, **kwargs)
                # Decoding generated Tokens
                for tok_seq in generated_tokens:
                    sequence = self.tokenizer.decode(tok_seq)
                    sequences.append(sequence)
                    
                sentences_list.append(sequences)

        return sentences_list
    

class Args:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data_path = 'train_preprocessed_corrupted.csv'
        self.test_data_path = 'test_preprocessed_corrupted.csv'
        self.checkpoint = None
        self.save_dir = "/"
        self.summary_dir = '/'
        self.device = 'cuda'
        self.model = 'gpt2-medium'
        self.max_length = 100
        self.batch_size = 32
        self.gradient_accumulation = 4
        self.learning_rate = 5.25e-5
        self.num_epochs = 1
        self.warmup_ratio = 0.002
        self.save_steps = 1000
        self.seed = 1234

args = Args()

# Initialize and build the finetuned GPT2 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpt_model = FinetuneGPT2(args) 
gpt_model.build_model(checkpoint_dir=None)  # Set checkpoint_dir if resuming training

train_dataset = QQPDataset(gpt_model.tokenizer, args.train_data_path,
                               max_length= 100,
                               load_noise_data=True,
                               device=device)

test_dataset = QQPDataset(gpt_model.tokenizer, args.test_data_path,
                               max_length= 100,
                               load_noise_data=True,
                               device=device)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        # Created a customized DataLoader with pin_memory set to False
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            pin_memory=False
        )

####################
# Training
####################
last_step = 0
batch_size = 16
for begin_loc in range(0, len(train_dataset), batch_size):
    last_step += 1

training_args = TrainingArguments(
    output_dir='/',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=args.gradient_accumulation,
    learning_rate=args.learning_rate,
    warmup_steps=10,
    weight_decay=0.01,
    save_steps=args.save_steps,
    eval_steps=args.save_steps,
    seed=args.seed,
    report_to = 'none',
#     should_save = False,
)

trainer = CustomTrainer(
    model=gpt_model.model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model()

##################################
# Saving trained model & Tokenizer
##################################

# Creating Checkpoint
os.makedirs("saved/", exist_ok=True)
model_path = os.path.join("saved/", "model")
tokenizer_path = os.path.join("saved/", "tokenizer")
gpt_model.model.save_pretrained(model_path)
gpt_model.tokenizer.save_pretrained(tokenizer_path)


#################################
# Testing Whole Test Data
#################################

def inference(args, gpt_model):
    
    sentences = []
    original = []
    with open(args.data_path) as f:
        reader = csv.reader(f)
        for corrupted, original_sentence in reader:
            sentences.append(corrupted)
            original.append(original_sentence)

    seq_list = gpt_model.generate_text(
        sentences,
        max_length=args.max_length,
        decoding=args.decoding,
        suffix='[SEP]'
    )
    ans = []
    for temp in seq_list:
        res = []
        for i,sample_sentence in enumerate(temp):
            
            sep_idx = sample_sentence.find('[SEP]')
            end_idx = sample_sentence.find('<|endoftext|>')
            if sep_idx != -1 and end_idx != -1:
                extracted_sentence = sample_sentence[sep_idx + 5:end_idx]
                res.append(extracted_sentence)
                
        ans.append(res)    
    return original, sentences, ans


class InferenceArgs:
    def __init__(self):
        self.data_path = 'test_preprocessed_corrupted.csv'
        self.checkpoint = None
        self.model ='gpt2-medium'
        self.device = 'cuda'
        self.max_length = 1024
        self.decoding = 'sampling'
        self.beam_size = 8
        self.top_k = 0
        self.p = 2
        self.temperature = 1.0
        self.num_generate = 1
        self.tag = ''
        self.debug = False
        self.seed = 1234
    
args = InferenceArgs()

original,corrupted, ans = inference(args, gpt_model)


##################################
# Testing for sentences from paper
##################################


def inference_for_given_sentence(args, gpt_model):
    original = ['how do you send a private message to someone you’re following on quora?',
                'do you believe donald trump can make america great again?',
                'if we see something in our dreams and it happens to come out true after few days, what does that mean?',
                'what do i gift my boyfriend for his birthday?','What to (4x+1) ? How do you factor this without square rooting?','Who is the nicest person you have ever met?']
    corrupted = ['send private message following quora ?','believe donald trump america great ?', 'dreams happens come out true after days, mean?', 'gift boyfriend birthday?','(4x+1)? factor without satisfying rooting?','ever met? individual nicest']

    seq_list = gpt_model.generate_text(
        corrupted,
        max_length=args.max_length,
        decoding=args.decoding,
        suffix='[SEP]'
    )
    ans = []
    for temp in seq_list:
        res = []
        for i,sample_sentence in enumerate(temp):
            
            sep_idx = sample_sentence.find('[SEP]')
            end_idx = sample_sentence.find('<|endoftext|>')
            if sep_idx != -1 and end_idx != -1:
                extracted_sentence = sample_sentence[sep_idx + 5:end_idx]
                res.append(extracted_sentence)
                
        ans.append(res)                
    return original, corrupted, ans


original_2, corrupted_2, ans_2 = inference_for_given_sentence(args, gpt_model)

for i, sentence in enumerate(original_2):
    print("input: ", sentence)
    
    print()
    print("corrupted: ", corrupted_2[i])
    print()
    for k, output in enumerate(ans_2[i]):
        print("op",k, ": ", output)
        print()
    print()
    print("------------------------------------------------------------")
    print()

####################################
# Saving [input, paraphrsed] to csv
####################################
output_csv_path = "output2.csv"

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # header row
    csvwriter.writerow(['Original Sentence', 'Paraphrase'])
    
    # Iterate over each original sentence and its paraphrases
    for i, sentence in enumerate(original_2):
        # Write each paraphrase with it's original
        for output in ans_2[i]:
            csvwriter.writerow([sentence, output])
            
print("Output saved to", output_csv_path)




###################
# Evaluation
###################

def calculate_contextual_similarity_hindi(sentence1, sentence2):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Tokenize and encode sentences
    inputs_sentence1 = tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
    inputs_sentence2 = tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)

    # Get BERT contexual embeddings for each sentences
    with torch.no_grad():
        outputs_sentence1 = model(**inputs_sentence1)
        outputs_sentence2 = model(**inputs_sentence2)

    # Calculate cosine sim between sentence embeddings
    similarity_score = torch.nn.functional.cosine_similarity(outputs_sentence1.last_hidden_state.mean(dim=1),
                                                             outputs_sentence2.last_hidden_state.mean(dim=1),
                                                             dim=1).item()
    return similarity_score

count = 0
li = []
for i, sentence in enumerate(original_2):
    temp = []
    count += 1
    if(count == 20):
        break
    print("input: ", sentence)
    print()
    print("corrupted: ", corrupted_2[i])
    print()
    for k, output in enumerate(ans_2[i]):
        
        score = calculate_contextual_similarity_hindi(sentence, output)
        temp.append((output, score))
        print("op",k, ": ", output, "score", score)
        print()
    print()
    print("------------------------------------------------------------")
    print()
    li.append(temp)

def get_score(item):
    return item[1]

# Print sorted list
for i, item in enumerate(li):
    item = sorted(item, key=get_score, reverse=True)
    print("input: ", original_2[i])
    print()
    print("corrupted: ", corrupted_2[i])
    print()
    for i, (output, score) in enumerate(item):
        print("Op",i + 1,": ", output, " Score:", score)
        print()
    print("------------------------------------------------------------")

