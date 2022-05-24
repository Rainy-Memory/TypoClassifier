import os
prefix = [
    # 'imdb_base', 
    'agnews_base', 
    # 'imdb_trNTY', 'agnews_trNTY', 'imdb_trTY', 'agnews_trTY', 
    # 'imdb_ptrTY_1', 'imdb_ptrTY_2', 'imdb_ptrTY_3', 'imdb_ptrTY_4', 'imdb_ptrNTY_1', 'imdb_ptrNTY_2', 'imdb_ptrNTY_3', 'imdb_ptrNTY_4', 
    # 'agnews_ptrTY_1', 'agnews_ptrTY_2', 'agnews_ptrTY_3', 'agnews_ptrTY_4', 'agnews_ptrNTY_1', 'agnews_ptrNTY_2', 'agnews_ptrNTY_3', 'agnews_ptrNTY_4',
    # 'imdb_4_epoch', 'agnews_4_epoch',
    # 'imdb_4_epoch_typo', 'agnews_4_epoch_typo',
]
for p in prefix:
    fu = os.system
    fu(f'python3 classifier.py {p} > bin/{p}.out')