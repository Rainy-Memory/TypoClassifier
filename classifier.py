import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import text_process

logging.set_verbosity_error()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def PrintState(text):
    def decorator(func):
        def wrapper(*args, **kwarg):
            text_dot = text + '...'
            text_end = text + ' finished'
            print(f'--------------- {text_dot:^45} ---------------')
            t = time.time()
            r = func(*args, **kwarg)
            t_min = (time.time() - t) / 60
            text_end += (f' in {t_min:.2f} minute.')
            print(f'--------------- {text_end:^45} ---------------')
            return r
        return wrapper
    return decorator

def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

def model_predict(model, device, dl):
    model.eval()
    all_logits = []
    for batch in dl:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    probs = nn.functional.softmax(all_logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds

loss_fn = nn.CrossEntropyLoss()

class BertClassifier(nn.Module):
    dir_name = 'model'
    bert_path = 'bert'
    inner_path = 'inner'
    classifier_path = 'classifier'

    def __init__(self, prefix, read_from, dout=4, fp_num=0): # fp: from pretrained
        super().__init__()
        self.prefix = prefix
        self.read_from = read_from
        # BERT output 768 dim vec
        din, in1, in2, h = 768, 300, 100, 50
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.inner0 = nn.Sequential(
            nn.Linear(din, in1),
            nn.ReLU()
        )
        self.inner1 = nn.Sequential(
            nn.Linear(in1, in2),
            nn.ReLU()
        )
        self.inner2 = nn.Sequential(
            nn.Linear(in2, h),
            nn.ReLU()
        )
        self.classifier = nn.Linear(h, dout)
        if fp_num >= 1:
            self.bert.load_state_dict(torch.load(f"{self.dir_name}/{self.read_from}_{self.bert_path}.pt"))
        if fp_num >= 2:
            self.inner0.load_state_dict(torch.load(f"{self.dir_name}/{self.read_from}_{self.inner_path}0.pt"))
        if fp_num >= 3:
            self.inner1.load_state_dict(torch.load(f"{self.dir_name}/{self.read_from}_{self.inner_path}1.pt"))
        if fp_num >= 4:
            self.inner2.load_state_dict(torch.load(f"{self.dir_name}/{self.read_from}_{self.inner_path}2.pt"))
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_CLS = output[0][:, 0, :]
        temp0 = self.inner0(last_hidden_state_CLS)
        temp1 = self.inner1(temp0)
        temp2 = self.inner2(temp1)
        logits = self.classifier(temp2)
        return logits

    def save(self):
        torch.save(self.bert.state_dict(), f"{self.dir_name}/{self.prefix}_{self.bert_path}.pt")
        torch.save(self.inner0.state_dict(), f"{self.dir_name}/{self.prefix}_{self.inner_path}0.pt")
        torch.save(self.inner1.state_dict(), f"{self.dir_name}/{self.prefix}_{self.inner_path}1.pt")
        torch.save(self.inner2.state_dict(), f"{self.dir_name}/{self.prefix}_{self.inner_path}2.pt")
        torch.save(self.classifier.state_dict(), f"{self.dir_name}/{self.prefix}_{self.classifier_path}.pt")

class TextClassification:
    def get_device(self):
        self.device = get_torch_device()

    @PrintState("Reading data")
    def read_data(self):
        seed_state = 42
        if self.dataset == 'agnews':
            self.train = pd.read_csv('ag_news/ag_news_train_typo.csv' if self.train_typo else 'ag_news/ag_news_train.csv').sample(frac=1, random_state=seed_state)[:40000]
            self.test = pd.read_csv('ag_news/ag_news_test_typo.csv' if self.test_typo else 'ag_news/ag_news_test.csv').sample(frac=1, random_state=seed_state)[:7600]
        else:
            self.train = pd.read_csv('imdb/imdb_train_typo.csv' if self.train_typo else 'imdb/imdb_train.csv').sample(frac=1, random_state=seed_state)[:40000]
            self.test = pd.read_csv('imdb/imdb_test_typo.csv' if self.test_typo else 'imdb/imdb_test.csv').sample(frac=1, random_state=seed_state)[:7600]
        self.x, self.y = self.train.text.values, self.train.label.values
        self.x_train, self.x_value, self.y_train, self.y_value = train_test_split(self.x, self.y, test_size=0.1, random_state=2022)
        self.max_len = 384
    
    @PrintState("Tokenizing data")
    def tokenize(self):
        self.train_input, self.train_mask = text_process.preprocessing_for_bert(self.x_train, self.max_len, self.tokenizer)
        self.value_input, self.value_mask = text_process.preprocessing_for_bert(self.x_value, self.max_len, self.tokenizer)

    @PrintState("Preprocessing torch")
    def torch_preprocess(self):
        # DataLoader = Sampler -> indices
        #            + Dataset -> data
        self.train_label, self.value_label = torch.tensor(self.y_train), torch.tensor(self.y_value)
        train_data = TensorDataset(self.train_input, self.train_mask, self.train_label)
        train_sampler = RandomSampler(train_data)
        self.train_dl = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        value_data = TensorDataset(self.value_input, self.value_mask, self.value_label)
        value_sampler = SequentialSampler(value_data)
        self.value_dl = DataLoader(value_data, sampler=value_sampler, batch_size=self.batch_size)

    @PrintState("Initializing model")
    def initializing_model(self):
        self.bert_classifier.to(self.device)
        self.optimizer = AdamW(self.bert_classifier.parameters(), lr=self.lr, eps=self.eps)
        total_step = len(self.train_dl) * self.epoch
        warmup_step = self.warmup_ratio * total_step
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

    def eval(self, model, val_dl):
        model.eval()
        val_loss, val_acc = [], []
        for batch in val_dl:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss)
            preds = torch.argmax(logits, dim=1).flatten()
            acc = (preds == b_labels).cpu().numpy().mean() * 100
            val_acc.append(acc)
        val_loss = np.mean(torch.tensor(val_loss).cpu().numpy())
        val_acc = np.mean(torch.tensor(val_acc).cpu().numpy())
        return val_loss, val_acc

    @PrintState("Training")
    def train_model(self, model, train_dl, optimizer, scheduler):
        show_batch_gap = len(train_dl) // 100
        # store_to_file_gap = len(train_dl) // 10
        for eph in range(1, self.epoch + 1):
            epoch_time, batch_time = time.time(), time.time()
            total_loss, batch_loss, batch_cnt = 0, 0, 0
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Test Loss':^10} | {'Test Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)
            model.train()
            for step, batch in enumerate(train_dl):
                batch_cnt += 1
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)
                model.zero_grad()
                logits = model(b_input_ids, b_attn_mask)
                loss = loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if (show_batch_gap != 0 and step % show_batch_gap == 0 and step != 0) or (step == len(train_dl) - 1):
                    time_elapsed = time.time() - batch_time
                    print(f"{eph:^7} | {step:^7} | {batch_loss / batch_cnt:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                    batch_loss, batch_cnt, batch_time  = 0, 0, time.time()
            avg_train_loss = total_loss / len(train_dl)
            print("-" * 70)
            test_loss, test_acc = self.eval(model, self.value_dl)
            time_elapsed = time.time() - epoch_time
            print(f"{eph:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {test_loss:^10.6f} | {test_acc:^9.2f} | {time_elapsed:^9.2f}")
            print(("-" * 70) + "\n\n")

    def evaluate_on_test(self):
        preds = model_predict(self.bert_classifier, self.device, self.value_dl)
        correct = (preds == self.y_value).sum()
        acc = (correct / len(self.y_value)) * 100
        print(f"Acc on test dataset: {acc:.2f}%")
        return acc

    def __init__(self, prefix, read_from=None, dataset='agnews', fp_num=0, train_typo=True, test_typo=True):
        self.prefix = prefix
        self.read_from = read_from
        self.dataset = dataset
        self.fp_num = fp_num
        self.train_typo = train_typo
        self.test_typo = test_typo

    # recommended hyper parameters:
    # batch size: 16 or 32
    # lr for adam: 5e-5, 3e-5 or 2e-5
    # epochs: 2, 3, or 4
    def main(self, lr=5e-5, eps=1e-8, epoch=2, batch_size=16, warmup_ratio=0.1):
        # set hyper parameters
        self.lr, self.eps = lr, eps
        self.epoch, self.batch_size, self.warmup_ratio = epoch, batch_size, warmup_ratio
        # init
        self.get_device()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.read_data()
        self.tokenize()
        self.torch_preprocess()
        self.bert_classifier = BertClassifier(prefix=self.prefix, read_from=self.read_from, dout=(4 if self.dataset == 'agnews' else 2), fp_num=self.fp_num)
        self.initializing_model()
        set_seed(42)
        self.train_model(self.bert_classifier, self.train_dl, self.optimizer, self.scheduler)
        self.bert_classifier.save()
        acc = self.evaluate_on_test()
        return acc

# name format: {dataset}_{train_typo}_{pretrain_typo}_{pretrained_level}.xxx
# dataset: agnews / imdb
# train_typo: trTY / trNTY
# pretrain_typo: ptrTY / ptrNTY
# pretrained_level: 
# dataset * 2, pre_typo * 2, level * 4

args = {
    # base case
    'imdb_base': {
        'prefix': 'imdb_base',
        'read_from': None,
        'dataset': 'imdb',
        'fp_num': 0,
        'train_typo': False,
        'test_typo': False,
    },
    'agnews_base': {
        'prefix': 'agnews_base',
        'read_from': None,
        'dataset': 'agnews',
        'fp_num': 0,
        'train_typo': False,
        'test_typo': False,
    },
    'imdb_trNTY': {
        'prefix': 'imdb_trNTY',
        'read_from': None,
        'dataset': 'imdb',
        'fp_num': 0,
        'train_typo': False,
    },
    'agnews_trNTY': {
        'prefix': 'agnews_trNTY',
        'read_from': None,
        'dataset': 'agnews',
        'fp_num': 0,
        'train_typo': False,
    },
    'imdb_trTY': {
        'prefix': 'imdb_trTY',
        'read_from': None,
        'dataset': 'imdb',
        'fp_num': 0,
        'train_typo': True,
    },
    'agnews_trTY': {
        'prefix': 'agnews_trTY',
        'read_from': None,
        'dataset': 'agnews',
        'fp_num': 0,
        'train_typo': True,
    },
    # imdb * 8
    'imdb_ptrTY_1': {
        'prefix': 'imdb_ptrTY_1',
        'read_from': 'agnews_trTY',
        'dataset': 'imdb',
        'fp_num': 1,
    },
    'imdb_ptrTY_2': {
        'prefix': 'imdb_ptrTY_2',
        'read_from': 'agnews_trTY',
        'dataset': 'imdb',
        'fp_num': 2,
    },
    'imdb_ptrTY_3': {
        'prefix': 'imdb_ptrTY_3',
        'read_from': 'agnews_trTY',
        'dataset': 'imdb',
        'fp_num': 3,
    },
    'imdb_ptrTY_4': {
        'prefix': 'imdb_ptrTY_4',
        'read_from': 'agnews_trTY',
        'dataset': 'imdb',
        'fp_num': 4,
    },
    'imdb_ptrNTY_1': {
        'prefix': 'imdb_ptrNTY_1',
        'read_from': 'agnews_trNTY',
        'dataset': 'imdb',
        'fp_num': 1,
    },
    'imdb_ptrNTY_2': {
        'prefix': 'imdb_ptrNTY_2',
        'read_from': 'agnews_trNTY',
        'dataset': 'imdb',
        'fp_num': 2,
    },
    'imdb_ptrNTY_3': {
        'prefix': 'imdb_ptrNTY_3',
        'read_from': 'agnews_trNTY',
        'dataset': 'imdb',
        'fp_num': 3,
    },
    'imdb_ptrNTY_4': {
        'prefix': 'imdb_ptrNTY_4',
        'read_from': 'agnews_trNTY',
        'dataset': 'imdb',
        'fp_num': 4,
    },
    # agnews * 8
    'agnews_ptrTY_1': {
        'prefix': 'agnews_ptrTY_1',
        'read_from': 'imdb_trTY',
        'dataset': 'agnews',
        'fp_num': 1,
    },
    'agnews_ptrTY_2': {
        'prefix': 'agnews_ptrTY_2',
        'read_from': 'imdb_trTY',
        'dataset': 'agnews',
        'fp_num': 2,
    },
    'agnews_ptrTY_3': {
        'prefix': 'agnews_ptrTY_3',
        'read_from': 'imdb_trTY',
        'dataset': 'agnews',
        'fp_num': 3,
    },
    'agnews_ptrTY_4': {
        'prefix': 'agnews_ptrTY_4',
        'read_from': 'imdb_trTY',
        'dataset': 'agnews',
        'fp_num': 4,
    },
    'agnews_ptrNTY_1': {
        'prefix': 'agnews_ptrNTY_1',
        'read_from': 'imdb_trNTY',
        'dataset': 'agnews',
        'fp_num': 1,
    },
    'agnews_ptrNTY_2': {
        'prefix': 'agnews_ptrNTY_2',
        'read_from': 'imdb_trNTY',
        'dataset': 'agnews',
        'fp_num': 2,
    },
    'agnews_ptrNTY_3': {
        'prefix': 'agnews_ptrNTY_3',
        'read_from': 'imdb_trNTY',
        'dataset': 'agnews',
        'fp_num': 3,
    },
    'agnews_ptrNTY_4': {
        'prefix': 'agnews_ptrNTY_4',
        'read_from': 'imdb_trNTY',
        'dataset': 'agnews',
        'fp_num': 4,
    },
    # train 4 epoch
    'agnews_4_epoch': {
        'prefix': 'agnews_4_epoch',
        'read_from': None, 
        'dataset': 'agnews',
        'fp_num': 0,
        'train_typo': False,
        'test_typo': True,
    },
    'imdb_4_epoch': {
        'prefix': 'imdb_4_epoch',
        'read_from': None, 
        'dataset': 'imdb',
        'fp_num': 0,
        'train_typo': False,
        'test_typo': True,
    },
    'agnews_4_epoch_typo': {
        'prefix': 'agnews_4_epoch_typo',
        'read_from': None, 
        'dataset': 'agnews',
        'fp_num': 0,
        'train_typo': True,
        'test_typo': True,
    },
    'imdb_4_epoch_typo': {
        'prefix': 'imdb_4_epoch_typo',
        'read_from': None, 
        'dataset': 'imdb',
        'fp_num': 0,
        'train_typo': True,
        'test_typo': True,
    },
}

if __name__ == '__main__':
    serial = sys.argv[1]
    arg = args[serial]
    print('arg:', arg)
    tc = TextClassification(**arg)
    tc.main(epoch=4)
