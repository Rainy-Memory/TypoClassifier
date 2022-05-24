print('Initializing...')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()
stops = np.load('stops.npy', allow_pickle=True)
stops = set(stops.tolist())

def text_preprocessing(s):
    s = s.lower()
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = " ".join([word for word in s.split() if word not in stops or word in ['not', 'can']])
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocessing_for_bert(data, tokenizer):
    input_ids, attention_masks = [], []
    for sent in data:
        encoded_sent = tokenizer(text=text_preprocessing(sent), add_special_tokens=True, max_length=384, truncation=True, padding='max_length', return_attention_mask=True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids, attention_masks = torch.tensor(input_ids), torch.tensor(attention_masks)
    return input_ids, attention_masks

class SentimentAnalysis(nn.Module):
    prefix = 'imdb_trTY'
    def __init__(self):
        super().__init__()
        din, in1, in2, h, dout = 768, 300, 100, 50, 2
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.inner0 = nn.Sequential(nn.Linear(din, in1), nn.ReLU())
        self.inner1 = nn.Sequential(nn.Linear(in1, in2), nn.ReLU())
        self.inner2 = nn.Sequential(nn.Linear(in2, h), nn.ReLU())
        self.classifier = nn.Linear(h, dout)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert.load_state_dict(torch.load(f'../model/{self.prefix}_bert.pt'))
        self.inner0.load_state_dict(torch.load(f'../model/{self.prefix}_inner0.pt'))
        self.inner1.load_state_dict(torch.load(f'../model/{self.prefix}_inner1.pt'))
        self.inner2.load_state_dict(torch.load(f'../model/{self.prefix}_inner2.pt'))
        self.classifier.load_state_dict(torch.load(f'../model/{self.prefix}_classifier.pt'))

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_CLS = output[0][:, 0, :]
        temp0 = self.inner0(last_hidden_state_CLS)
        temp1 = self.inner1(temp0)
        temp2 = self.inner2(temp1)
        logits = self.classifier(temp2)
        return logits

def eval(model: SentimentAnalysis, text: str) -> bool:
    input_ids, attn_masks = preprocessing_for_bert([text], model.tokenizer)
    with torch.no_grad():
        logits = model(input_ids, attn_masks)
    preds = torch.argmax(logits, dim=1).flatten()
    return bool(preds[0] == torch.tensor(1))

def main():
    sa = SentimentAnalysis()
    sa.eval()
    print('Welcome to this naive sentiment analysis demo with robust to typo! Input any text and see whether the prediction is correct :)')
    print('Input "[q]" to quit program.')
    while True:
        text = input('Input text to analysis: ')
        if text == '[q]':
            return
        res = eval(sa, text)
        print('This sentence is ' + ('positive :)' if res else 'negative :('))

if __name__ == '__main__':
    main()