import pickle
import random
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sk_text
import torch
from torch import nn


class EmbedHistSimilarity(object):
    def __init__(self, idx_file, emb_file):
        with open(idx_file, 'rb') as f:
            self.word_to_idx: dict = pickle.load(f)
        embed_arr = np.load(emb_file)
        self.embed_size = embed_arr.shape[1]
        self.embed = nn.Embedding(*embed_arr.shape)
        self.embed.weight.data.copy_(torch.from_numpy(embed_arr))
        cv = sk_text.CountVectorizer(analyzer='word', stop_words='english', vocabulary=self.word_to_idx.keys())
        self.tok = cv.build_tokenizer()

    def text_to_embed_hist(self, t: str):
        idx_ = [self.word_to_idx[w.lower()] for w in self.tok(t) if w.lower() in self.word_to_idx]
        if not idx_:
            return torch.zeros(self.embed_size)
        idxs = torch.tensor(idx_, dtype=torch.long)
        e = self.embed(idxs)
        return e.mean(0)

    def sim(self, x, ys):
        x_hist = self.text_to_embed_hist(x)
        ys_hist = map(self.text_to_embed_hist, ys)
        cos = nn.CosineSimilarity(dim=0)
        return [cos(x_hist, yh).item() for yh in ys_hist]


if __name__ == '__main__':
    df = pd.read_json('apps.json.gz', orient='index', convert_axes=False)
    descriptions = df['description']
    ts = random.choices(descriptions, k=5)
    t1 = ts[0]
    ts = ts[1:]
    ehs = EmbedHistSimilarity('word_to_idx.pickle', 'embed.npy')
    res = ehs.sim(t1, ts)
    print(res)
