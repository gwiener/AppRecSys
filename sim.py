import pickle
import random
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sk_text
import torch
from torch import nn

with open('word_to_idx.pickle', 'rb') as f:
    word_to_idx: dict = pickle.load(f)
embed_arr = np.load('embed.npy')
embed_size = embed_arr.shape[1]
embed = nn.Embedding(*embed_arr.shape)
embed.weight.data.copy_(torch.from_numpy(embed_arr))
df = pd.read_json('apps.json.gz', orient='index', convert_axes=False)
descriptions = df['description']

cv = sk_text.CountVectorizer(analyzer='word', stop_words='english', vocabulary=word_to_idx.keys())
tok = cv.build_tokenizer()


def text_to_embed_hist(t: str):
    idx_ = [word_to_idx[w.lower()] for w in tok(t) if w.lower() in word_to_idx]
    if not idx_:
        return torch.zeros(embed_size)
    idxs = torch.tensor(idx_, dtype=torch.long)
    e = embed(idxs)
    return e.mean(0)


def embed_hist_sim(x, ys):
    xh = text_to_embed_hist(x)
    yhs = map(text_to_embed_hist, ys)
    cos = nn.CosineSimilarity(dim=0)
    return [cos(xh, yh).item() for yh in yhs]


ts = random.choices(descriptions, k=5)
t1 = ts[0]
ts = ts[1:]
res = embed_hist_sim(t1, ts)
print(res)
