import tqdm
import random
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sk_text

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

vocab_size = 500
n_gram_size = 3
embedding_size = 10
n_epochs = 8
batch_size = 32
sample_rate=0.9

df = pd.read_json('apps.json.gz', orient='index', convert_axes=False)
cv = sk_text.CountVectorizer(analyzer='word', stop_words='english', min_df=0.01, max_df=0.8, max_features=vocab_size)
tok = cv.build_tokenizer()
text = df['description']
cv.fit(text)
word_to_idx = cv.vocabulary_
samples = [
    [word_to_idx[w.lower()] for w in tok(sentence) if w.lower() in word_to_idx]
    for sentence in text
]
samples = list(filter(lambda x: len(x) >= n_gram_size, samples))
samples = random.choices(samples, k=int(len(samples)*sample_rate))
print(len(samples))
n_grams = [
    n_gram
    for sample in samples
    for n_gram in [sample[i:i + n_gram_size] for i in range(len(sample) - n_gram_size + 1)]
]
print(len(n_grams))
n_batches = len(n_grams) // batch_size
print(n_batches)
print(n_batches*batch_size)


def batch_iter():
    for b in range(n_batches):
        batch = [n_grams[i] for i in range(b, n_batches*batch_size, n_batches)]
        ctxs = [x[:-1] for x in batch]
        targets = [x[-1] for x in batch]
        yield ctxs, targets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size=128):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view((batch_size, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


loss_function = nn.NLLLoss()
model = NGramLanguageModeler(vocab_size, embedding_size, n_gram_size - 1)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    total_loss = 0
    t_bar = tqdm.tqdm(batch_iter(), total=n_batches, postfix=dict(loss=0))
    for batch in t_bar:
        curr_contexts, curr_targets = batch
        context_idxs = torch.tensor(curr_contexts, dtype=torch.long).to(device)
        model.zero_grad()
        log_probs = model(context_idxs)
        target_idx = torch.tensor(curr_targets, dtype=torch.long).to(device)
        loss = loss_function(log_probs, target_idx)
        loss.backward()
        optimizer.step()
        step_loss = loss.cpu().item()
        total_loss += step_loss
        t_bar.set_postfix(loss=total_loss)

embeddings = model.embeddings.cpu().weight.data.numpy()
np.save('embed.npy', embeddings)
