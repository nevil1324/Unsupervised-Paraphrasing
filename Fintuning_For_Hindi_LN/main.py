from data_loader import HindiPeraphraseDataset
from preprocess import LoadDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, random_split
import gc
from transformers import Trainer, TrainingArguments
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

def train_model(train_dataset, val_dataset):
    training_args = TrainingArguments(
    output_dir='./result',
    num_train_epochs=5,
    logging_steps=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10, 
    weight_decay=0.05,
    optim="adafactor",
    learning_rate=5e-6,
    save_steps=5000,
    logging_dir='./logs',
    gradient_checkpointing=True,
    report_to = 'none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

def corrupt(sen):
    random.shuffle(sen)
    return sen

def print_paraphrases(model,tokenizer,sentence):
#     print(sentence)
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer("hi",remove_nuktas=False)
    corrupted = " ".join(corrupt(indic_tokenize.trivial_tokenize(normalizer.normalize(sentence.strip()))))
    corrupt_in = "[BOS]" + corrupted + "[SEP]"
#     print(corrupt_in)
    generated = tokenizer(corrupt_in, return_tensors="pt").input_ids.cuda()
    sample_outputs = model.generate(generated, do_sample=True, top_k=20,
        top_p=0.975, temperature=1.0,
        no_repeat_ngram_size=10, num_return_sequences=20,max_length=128)
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output).split("[SEP]")[1].split("[EOS]")[0]))

def test_model(model,tokenizer,sentences):
    torch.cuda.empty_cache()
    hindi_stopwords = {
    "एक", "एवं", "यह", "इस", "के", "का", "की", "को", "में", "है", "हैं", "कर", "किया", "किए", "करते", "करना", "किसी", "गया", "जाता", "जाती", "जाते", "साथ", "अपने", "हुआ", "होता", "होती", "होते", "वाले", "वह", "वहाँ", "जैसा", "जिसका", "जिसकी", "जिसके", "जिनका", "जिनकी", "जिनके", "तथा", "उसके", "उसका", "उसकी", "उनके", "उनका", "उनकी", "उनको", "कुछ", "इसका", "इसकी", "इसके", "सभी", "अगर", "इसमें", "उनका", "उनकी", "उनके", "जैसे", "जिसमें", "तिन्हों", "तिन्हें", "पहले", "बाद", "मानो", "अंदर", "भीतर", "पूरे", "खुद", "आप", "अब", "जब", "जहाँ", "जितना", "जितने", "तब", "वहीं", "हुआ", "होता", "होती", "वाला", "वाली", "वाले"
    }
    for sen in sentences:
        sen = sen.replace("’", "")
        sen = sen.replace("'", "")
        sen = sen.replace("-", " ")
        words = sen.split()
        filtered_words = [word for word in words if word not in hindi_stopwords]
        sen = ' '.join(filtered_words)
        print()
        print('Original Sentence: ',sen)
        print("Paraphases:" )
        print_paraphrases(model,tokenizer,sen)


sentences =["मैं अपनी मां के साथ बाजार के लिए निकला",
   "मैं फल खाता हूँ",
   "मैं चॉकलेट खरीदना चाहता हूं ताकि मैं पढ़ाई कर सकूं",
   "मैं अपने घर जा रहा हूँ",
   "वह बहुत सुन्दर है",
   "वह बड़े ही समय से इस नगर में अपने अध्ययन को समर्पित कर रहा है, और उसकी मेहनत और निष्ठा का परिणाम यह है कि वह अब अपने क्षेत्र में एक प्रमुख नाम बन चुका है।",
   "वह उस नाटक में प्रमुख भूमिका निभाने के लिए बहुत मेहनत कर रहा है।",
   "जीवन में सफलता पाने के लिए प्रयास करो, न कि सफलता की तलाश में भटको। ",
    "वह अपने सपनों को पूरा करने के लिए पूरी ताकत लगाता है।",
    "खुशी वह है जो दूसरों को खुश देखने में मिलती है।",
    "सपने वो नहीं होते जो हम सोते समय देखते हैं, सपने वो होते हैं जो हमें सोने नहीं देते। ",
    "जीवन में सफलता का सबसे बड़ा रहस्य उसमें लगातार मेहनत करना है। ",
    "संघर्ष के बिना कोई भी महान नहीं बन सकता।",
    "संघर्ष के माध्यम से ही हमें अपने असली योग्यता का पता चलता है। ",
    "अपने सपनों को पूरा करने के लिए हमें उनकी महत्ता समझनी चाहिए, न कि उनके दुष्प्रभाव। ",
    "अपने मन का नहीं, अपने कर्मों का नतीजा मिलता है। ",
    "अगर आपका आज परिश्रम से बढ़िया है, तो आपका कल स्वयं आप बेहतर होगा।",
    "कठिनाईयों से डरना नहीं, उनसे लड़ना सीखो।",
    "अपने लक्ष्य को हासिल करने के लिए हमें कभी भी हार नहीं माननी चाहिए।",
    "विश्वास रखें, आशा रखें, और मेहनत करें। ",
    "बदलाव वह नहीं जो हमें डराए, बल्कि बदलाव वह है जिसे हम स्वीकार करते हैं।",
    "सफलता उन्हें मिलती है जो हार नहीं मानते।"
    ]

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
    model = GPT2LMHeadModel.from_pretrained("sberbank-ai/mGPT").cuda()
    print(len(tokenizer))
    tokenizer.add_special_tokens({"sep_token": '[SEP]', "bos_token":'[BOS]'})
    tokenizer.pad_token = '[EOS]'
    print(tokenizer.all_special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    ds = LoadDataset()
    sentences=ds.getdata()
    dataset = HindiPeraphraseDataset(sentences,tokenizer)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    print(len(train_dataset),len(val_dataset))

    gc.collect()
    torch.cuda.empty_cache()

    train_model(train_dataset, val_dataset)

    test_model(model,tokenizer,sentences)