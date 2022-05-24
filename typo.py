import pandas as pd
import random

def typo(word):
    if len(word) <= 2:
        return word[0] + word
    type = random.randint(0, 5)
    pos = random.randint(0, len(word) - 2)
    if type == 0: # duplicate
        return word[:pos] + word[pos] + word[pos:]
    elif type == 1: # delete
        return word[:pos] + word[pos + 1:]
    else: # change order
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]

def make_typo(data: pd.DataFrame):
    # old = data.copy(deep=True)
    for i, row in data.iterrows():
        s = row.text
        ls = s.split()
        ss = ''
        tar = random.randint(0, len(ls) - 1)
        while not ls[tar].isalpha():
            tar = random.randint(0, len(ls) - 1)
        for idx, word in enumerate(ls):
            ss += (word if idx != tar else typo(word)) + ' '
        data.at[i, 'text'] = ss
    # return pd.concat([data, old], axis=0).reset_index(drop=True)
    # return data

for dir in ['ag_news', 'imdb']:
    for case in ['test']:
        print(dir, case)
        a = pd.read_csv(f'{dir}/{dir}_{case}.csv')
        make_typo(a)
        a.to_csv(f'{dir}/{dir}_{case}_typo.csv')