"""
Create t-SNE visualisations.
"""

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

import sys
PATH_TO_REPSEVAL = "../repseval/src"
sys.path.insert(0, PATH_TO_REPSEVAL)
from evaluate import evaluate_embed_matrix
from wordreps import WordReps

def select_words(vocab_fname, sentiwordnet_fname, selected_fname):
    """
    Select highly positive and negative words that are in the vocabulary.
    """
    vocab = []
    with open(vocab_fname) as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())
    print("Length of vocabulary =", len(vocab))

    sent_words = {}
    with open(sentiwordnet_fname) as sent_file:
        for line in sent_file:
            if not line.startswith('#'):
                p = line.strip().split()
                score = float(p[1])
                if score != 0:
                    sent_words[p[0].split('#')[0]] = score
    print("Total no. of words with pos/neg sentiment =", len(sent_words))

    h = {} # selected words from the vocabulary that have pos/neg sentiment
    for w in vocab:
        if w in sent_words:
            h[w] = sent_words[w]
    
    print("Overlapping sentiment words =", len(h))
    L = list(h.items())
    L.sort(key=lambda x: -x[1])
    pos_words = L[:100]
    neg_words = L[-100:]
    neg_words.reverse()

    with open(selected_fname, 'w') as F:
        for w in pos_words:
            F.write("1 %s\n" % w[0])
        for w in neg_words:
            F.write("-1 %s\n" % w[0])
    pass

def guess_dim(fname):
    with open(fname) as F:
        l = F.readline().split()
    return len(l) - 1

def plot_tsne(embed_fname, sel_fname):
    labels = []
    tokens = []

    sel_words = {}
    with open(sel_fname) as F:
        for line in F:
            p = line.strip().split()
            sel_words[p[1]] = int(p[0])

    WR = WordReps()
    d = guess_dim(embed_fname)
    WR.read_model(embed_fname, d)

    for word in WR.vects:
        if word in sel_words:
            tokens.append(WR.vects[word])
            labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    for i in range(len(x)):
        if sel_words[labels[i]] == 1:
            plt.scatter(x[i],y[i], c='blue')
        else:
             plt.scatter(x[i],y[i], c='red')
        #plt.xlim(xmin, xmax)
        #plt.ylim(ymin, ymax)
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=18)
    plt.xlim(-80,80)
    #plt.ylim(-10,10)
    plt.savefig("fig.png")
    pass



if __name__ == "__main__":
    #select_words("./data/source_embeddings/text8/vocab", "./data/SentiWords_1.1.txt", "./data/pos-neg.words")
    plot_tsne("dw.embeds", "./data/pos-neg.words")