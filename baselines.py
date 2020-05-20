"""
Calls/implements baseline methods for meta embedding for comparisons.
"""

import numpy as np 
from sklearn.utils.extmath import randomized_svd

import pandas as pd
from tabulate import tabulate

import sys
PATH_TO_REPSEVAL = "../repseval/src"
sys.path.insert(0, PATH_TO_REPSEVAL)
from evaluate import evaluate_embed_matrix, evaluate_embeddings
from wordreps import WordReps
import glob
from GlobalME import GLME

def load_data(vocab_fname):
    """
    Load the vocabulary and cooccurrence marix. Only consider top cutoff frequent words
    from the vocabulary.

    Parameters
    -----------
        vocab_fname : string
            name of the vocabulary file.
    
    Returns
    --------
        vocab : dict
            A dictionary mapping words to their row indices in the cooccurrece matrix.
        
        reverse_vocab : dict
            A dictionary mapping row indices in the cooccurrence matrix to the words.
    """
    vocab = {} # word:int
    reverse_vocab = {} # int:word
    with open(vocab_fname) as vocab_file:
        count = 0
        for line in vocab_file:
            word = line.strip().split()[0]
            vocab[word] = count
            reverse_vocab[count] = word
            count += 1
    
    return vocab, reverse_vocab

def load_source(fname):
    with open(fname, "rb") as F:
        return np.load(F)

def svd_baseline(df, sources, vocab, k):
    """
    Concatenate all sources and apply SVD to reduce dimensionality to k.
    """
    M = np.concatenate(sources, axis=1)
    print("concatenated M = ", M.shape)
    U, A, VT = randomized_svd(M, n_components=k, random_state=None)
    #print(U.shape)
    #print(A.shape)
    #print(VT.shape)
    R = U[:,:k] @ np.diag(A[:k])
    WR = WordReps()
    WR.load_matrix(R, vocab)
    res = evaluate_embed_matrix(WR, mode="all")
    res["k"] = k
    df = df.append(pd.DataFrame(res, index=["svd"]))
    return df

def avg_baseline(df, sources, vocab):
    """
    Average the source embeddings.
    """
    M = np.mean(sources, axis=0)
    WR = WordReps()
    WR.load_matrix(M, vocab)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode="all"), index=["avg"]))
    return df

def write_embeds(M, fname, vocab):
    reverse_vocab = {}
    for word in vocab:
        reverse_vocab[vocab[word]] = word
    
    with open(fname, 'w') as F:
        for i in range(M.shape[0]):
            F.write("%s %s\n" % (reverse_vocab[i], " ".join([str(x) for x in M[i,:]])))
    pass

def evaluate_me(me_fname, dim):
    """
    Evaluates meta embeddings created by external libraries such as AE.
    """
    df = pd.DataFrame()
    df = df.append(pd.DataFrame(evaluate_embeddings(me_fname, dim, mode="all"), index=["AE"]))
    df.to_csv("baseline-res.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    pass     

def globally_linear_me(sources, vocab, df, d, lr):
    M_list, A = GLME([A.T for A in sources], d, lr)
    WR = WordReps()
    WR.load_matrix(A.T, vocab)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode="all"), index=["GLME %d %f" % (d,lr) ]))
    return df


def process():
    """
    check whether all dictionaries are the same. 
    Then save one of those dictionaries
    save source embeddings for text8, small and 1G separately
    """
    prefix = "./data/source_embeddings/text8"
    source_fnames = ["glove.npz", "word2vec.npz", "lsa.npz"]

    vocab = {}
    count = 0
    with open("%s/vocab" % prefix) as F:
        for line in F:
            vocab[line.strip()] = count
            count += 1

    sources = []
    for fname in source_fnames:
        source = load_source("%s/%s" % (prefix,fname))
        print(fname, source.shape)

        # write the sources for LLE/AEME processing
        write_embeds(source, fname.split(".")[0]+ ".embed", vocab)
        sources.append(source)
    
    
    df = pd.DataFrame()

    # svd-baseline
    #for k in range(50, 900, 50):
    #    df = svd_baseline(df, sources, vocab, k)

    # avg baseline
    #df = avg_baseline(df, sources, vocab)

    # Globally Linear meta-embedding
    for d in [100, 200, 300, 400, 500, 600, 700, 800]:
        for lr in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            df = globally_linear_me(sources, vocab, df, d, lr)


    # evaluate LLE
    #fnames = glob.glob("../LLE-MetaEmbed/work/meta-embeds/*")
    #for fname in fnames:
    #    print(fname)
    #    k = int(fname.split('+k=')[1])
    #    ind = fname.split("/")[-1]
    #    df = df.append(pd.DataFrame(evaluate_embeddings(fname, k, mode="all"), index=[ind]))        

    df.to_csv("baseline-res.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    pass

if __name__ == "__main__":
    process()
    #evaluate_me("../AutoencodedMetaEmbedding/aeme/work/res.txt", 300)

