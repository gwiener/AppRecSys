import tqdm
import pickle
import random
import numpy as np
import pandas as pd
import configargparse
import sklearn.feature_extraction.text as sk_text

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def rand_batches(n_batches):
    res = list(range(n_batches))
    random.shuffle(res)
    return res


def batch_iter(samples, batch_size, n_batches):
    for b in rand_batches(n_batches):
        batch = [samples[i] for i in range(b, n_batches*batch_size, n_batches)]
        ctxs = [x[:-1] for x in batch]
        targets = [x[-1] for x in batch]
        yield ctxs, targets


def cbow_batch_iter(samples, sample_size, batch_size, n_batches):
    for b in rand_batches(n_batches):
        batch = [samples[i] for i in range(b, n_batches*batch_size, n_batches)]
        ctxs = [[x[i] for i in range(sample_size) if i != j] for x in batch for j in range(sample_size)]
        targets = [x[j] for x in batch for j in range(sample_size)]
        yield ctxs, targets


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, batch_size, hidden_size):
        super(NGramLanguageModeler, self).__init__()
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view((self.batch_size, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def main(
    vocab_size=0,
    n_gram_size=0,
    embed_size=0,
    batch_size=0,
    hidden_size=0,
    num_epochs=0,
    learn_rate=0,
    momentum=0,
    lr_decay=0,
    weight_decay=0,
    sample_rate=0,
    optimizer='',
    cbow=False,
    min_df=0,
    max_df=0,
    idx_file='',
    emb_file='',
    in_file=''
):
    df = pd.read_json(in_file, orient='index', convert_axes=False)
    cv = sk_text.CountVectorizer(
        analyzer='word', stop_words='english',
        min_df=min_df, max_df=max_df,
        max_features=vocab_size
    )
    tok = cv.build_tokenizer()
    text = df['description']
    cv.fit(text)
    word_to_idx = cv.vocabulary_
    with open(idx_file, 'wb') as f:
        pickle.dump(word_to_idx, f)
    samples = [
        [word_to_idx[w.lower()] for w in tok(sentence) if w.lower() in word_to_idx]
        for sentence in text
    ]
    samples = list(filter(lambda x: len(x) >= n_gram_size, samples))
    if sample_rate < 1.0:
        samples = random.choices(samples, k=int(len(samples) * sample_rate))
    print(len(samples))
    n_grams = [
        n_gram
        for sample in samples
        for n_gram in [sample[i:i + n_gram_size] for i in range(len(sample) - n_gram_size + 1)]
    ]
    print(len(n_grams))
    n_batches = len(n_grams) // batch_size
    print(n_batches)
    print(n_batches * batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_function = nn.NLLLoss()
    actual_batch_size = batch_size*n_gram_size if cbow else batch_size
    model = NGramLanguageModeler(vocab_size, embed_size, n_gram_size-1, actual_batch_size, hidden_size)
    model = model.to(device)
    optimizer_func: optim.Optimizer = None
    if optimizer == 'sgd':
        optimizer_func = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum, weight_decay=weight_decay)
    if optimizer == 'ada':
        optimizer_func = optim.Adagrad(model.parameters(), lr=learn_rate, lr_decay=lr_decay, weight_decay=weight_decay)
    if optimizer == 'adam':
        optimizer_func = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    assert optimizer_func, "%s is not a valid optimizer name" % optimizer
    for epoch in range(num_epochs):
        total_loss = 0
        batch_gen = (
            cbow_batch_iter(n_grams, n_gram_size, batch_size, n_batches) if cbow
            else batch_iter(n_grams, batch_size, n_batches)
        )
        t_bar = tqdm.tqdm(batch_gen, total=n_batches, postfix=dict(loss=0))
        for batch in t_bar:
            curr_contexts, curr_targets = batch
            context_idxs = torch.tensor(curr_contexts, dtype=torch.long).to(device)
            model.zero_grad()
            log_probs = model(context_idxs)
            target_idx = torch.tensor(curr_targets, dtype=torch.long).to(device)
            loss = loss_function(log_probs, target_idx)
            loss.backward()
            optimizer_func.step()
            step_loss = loss.cpu().item()
            total_loss += step_loss
            t_bar.set_postfix(loss=total_loss)
    embeddings = model.embeddings.cpu().weight.data.numpy()
    np.save(emb_file, embeddings)


if __name__ == '__main__':
    parser = configargparse.ArgParser(default_config_files=['./train.conf'])
    parser.add('-c', '--conf', is_config_file=True)
    parser.add('-vs', '--vocab-size', type=int)
    parser.add('-ns', '--n-gram-size', type=int)
    parser.add('-es', '--embed-size', type=int)
    parser.add('-bs', '--batch-size', type=int)
    parser.add('-hs', '--hidden-size', type=int)
    parser.add('-ne', '--num-epochs', type=int)
    parser.add('-lr', '--learn-rate', type=float)
    parser.add('--momentum', type=float)
    parser.add('--lr-decay', type=float)
    parser.add('--weight-decay', type=float)
    parser.add('--sample-rate', type=float)
    parser.add('-o', '--optimizer', type=str, choices=['sgd', 'ada','adam'])
    parser.add('--cbow', dest='cbow', action='store_true', default=False)
    parser.add('--min-df', type=float)
    parser.add('--max-df', type=float)
    parser.add('--idx-file', type=str)
    parser.add('--emb-file', type=str)
    parser.add('--in-file', type=str)
    args = parser.parse_args()
    kw_args = vars(args)
    kw_args.pop('conf')
    main(**kw_args)