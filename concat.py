"""
This file implements the PIP-loss-based concatenated meta-embedding method.

Danushka Bollegala
1st May 2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import scipy.sparse as sp
import pandas as pd
from tabulate import tabulate
import yaml

from matrix.signal_matrix_factory import SignalMatrixFactory
from matrix.PIP_loss_calculator import MonteCarloEstimator
from utils.svd import svd

import numpy as np

import sys
PATH_TO_REPSEVAL = "../repseval/src"
sys.path.insert(0, PATH_TO_REPSEVAL)

from evaluate import evaluate_embed_matrix
from wordreps import WordReps


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

def load_binary_matrix(fname, vocab_size):
    print("Loading {0} ...".format(fname))
    with open(fname, 'rb') as F:
        i, j, val  = zip(*np.fromfile(F, dtype=np.dtype([('i', np.int32), ('j', np.int32), ('val', np.double)])))
        M = sp.csr_matrix((val, (i,j)), shape=(vocab_size + 1, vocab_size + 1)).toarray()
        return M[1:,1:]
    raise ValueError


def create_signal_matrix(algorithm, vocab, reverse_vocab, M, A, B):
    factory = SignalMatrixFactory()
    signal_matrix = factory.produce(algorithm)
    signal_matrix.set_params(vocab, reverse_vocab, M, A, B)
    path = signal_matrix.param_dir
    signal_matrix.estimate_signal()
    signal_matrix.estimate_noise()
    signal_matrix.export_estimates()
    return path, signal_matrix

def estimate_pip(path):
    pip_calculator = MonteCarloEstimator()
    pip_calculator.get_param_file(path, "estimates.yml")
    pip_calculator.estimate_signal()
    pip_calculator.estimate_pip_loss()
    pip_calculator.plot_pip_loss()
    return pip_calculator    

def concat(source_matrices, weights_list):
    """
    Concatenates the given source embedding matrices. Can be used to perform both source-specific and
    dimension-specific concatenation.
    
    Parameters
    ----------
    source_matrices: a list of numpy.ndarrays
                    each representing n x k embeddng matrix. The vocabulary of the words n must be equal in all embedding matrices and their
                    dimensionalities k can be different.
        
    weights_list: diagonal weights (numpy.ndarrays) for each source.
                In the case of source-specific weighting all weights for a source will be equal, whereas for dimension-specific weighting they will be different.
    
    Returns
    -------
    concat: a single numpy.ndarray 
            with the same n rows and the number of columns equals the sum of source dimensionalities.
    """
    assert(len(source_matrices) == len(weights_list))
    return np.concatenate([source_matrices[i] @ np.diag(weights_list[i]) for i in range(len(source_matrices))], axis=1)


def get_source_weighted_concat_coef(lmdas, myus, k):
    """
    Compute the concatenation coefficient under source-weighted concatenation.
    
    Parameters
    ----------
    lambdas : numpy.array
        spectrum of the signal matrix.
        
    myus :  numpy.array
        spectrum of the embedding matrix.
        
    k : int
        rank of the embedding matrix.
        
    Returns
    --------
    c : float
        the concatenation coefficient.
    """
    return np.dot(lmdas[:k], myus[:k]) / np.dot(myus[:k], myus[:k])


def get_dimension_weighted_concat_coef(lmdas, myus, k):
    """
    Compute the concatenation coefficient under dimension-weighted concatenation.
    
    Parameters
    ----------
    lambdas : numpy.array
        spectrum of the signal matrix.
        
    myus :  numpy.array
        spectrum of the embedding matrix.
        
    k : int
        rank of the embedding matrix.
        
    Returns
    --------
    c : numpy.array
        the concatenation coefficient.
    """
    return lmdas[:k] / myus[:k]
    

def process():
    settings = ["glove", "word2vec", "lsa"]

    with open(sys.argv[1]) as cfg_file:
        cfg = yaml.load(cfg_file)
    vocab_size = cfg["vocab_size"]
    vocab, reverse_vocab = load_data(cfg["vocab"])

    M = load_binary_matrix(cfg["M_fname"], vocab_size)
    A = load_binary_matrix(cfg["A_fname"], vocab_size)
    B = load_binary_matrix(cfg["B_fname"], vocab_size)

    path = {}
    signal_matrix = {}
    pip_calculator = {}

    for algorithm in settings:
        print(algorithm)
        path[algorithm], signal_matrix[algorithm]= create_signal_matrix(algorithm, vocab, reverse_vocab, M, A, B)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])     

    k = {}
    sources = []
    df = pd.DataFrame()
    for algo in settings:
        print(algo)
        k[algo] = np.argmin(pip_calculator[algo].estimated_pip_loss)
        #k[algo] = 300
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ np.diag(signal_matrix[algo].spectrum[:k[algo]])
        sources.append(source_mat)      
        WR = WordReps()
        WR.load_matrix(source_mat, vocab)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode="lex"), index=[algo]))
    
    print("\nUnweighted concatenation...")
    weights_list = [np.ones(k[algo]) for algo in settings]
    M1 = concat(sources, weights_list)
    WR1 = WordReps()
    WR1.load_matrix(M1, vocab)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR1, mode="lex"), index=["unweighted"]))

    # source-weighted concatenation
    print("\nSource weighted concatenation...")
    weights_list = []
    for algo in settings:
        c = get_source_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            np.array(pip_calculator[algo].estimated_signal), 
                                           k[algo])
        weights_list.append(c * np.ones(k[algo]))
    M2 = concat(sources, weights_list)
    WR2 = WordReps()
    WR2.load_matrix(M2, vocab)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR2, mode="lex"), index=["source-weighted"]))

    print("\nDimension weighted concatenation...")
    weights_list = []
    for algo in settings:
        c = get_dimension_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            np.array(pip_calculator[algo].estimated_signal), 
                                           k[algo])
        weights_list.append(c)
    M3 = concat(sources, weights_list)
    WR3 = WordReps()
    WR3.load_matrix(M3, vocab)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR3, mode="lex"), index=["dim-weigted"]))

    # save and display results
    df.to_csv(cfg["res_fname"])
    print(tabulate(df, headers='keys', tablefmt='psql'))

def test_binary():
    with open("./data/secondcooc_10k.bin", "rb") as F:
        i, j, val  = zip(*np.fromfile(F, dtype=np.dtype([('i', np.int32), ('j', np.int32), ('val', np.double)])))
        print(len(i))
        print(len(j))
        print(len(val))
        M = sp.csr_matrix((val, (i,j)), shape=(10001,10001))
        print(M.shape)
        print(M[23,46])
        print(M[46,23])
    pass


if __name__ == "__main__":
    process()
    #test_binary()