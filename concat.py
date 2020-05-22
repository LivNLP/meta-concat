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
import yaml

from matrix.signal_matrix_factory import SignalMatrixFactory
from matrix.PIP_loss_calculator import MonteCarloEstimator
from utils.tokenizer import SimpleTokenizer
from utils.reader import ReaderFactory

import pandas as pd
from tabulate import tabulate

import numpy

import sys
PATH_TO_REPSEVAL = "../repseval/src"
sys.path.insert(0, PATH_TO_REPSEVAL)
from evaluate import evaluate_embed_matrix
from wordreps import WordReps

def create_signal_matrix(corpus_fname, model_config, algorithm):
    with open(model_config, "r") as f:
        cfg = yaml.load(f)

    reader = ReaderFactory.produce(corpus_fname[-3:])
    data = reader.read_data(corpus_fname)
    tokenizer = SimpleTokenizer()
    indexed_corpus = tokenizer.do_index_data(data,
        n_words=cfg.get('vocabulary_size'),
        min_count=cfg.get('min_count'))

    factory = SignalMatrixFactory(indexed_corpus)

    signal_matrix = factory.produce(algorithm)
    path = signal_matrix.param_dir
    signal_matrix.inject_params(cfg)
    signal_matrix.estimate_signal()
    signal_matrix.estimate_noise()
    signal_matrix.export_estimates()
    return cfg, path, signal_matrix, tokenizer

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
    return numpy.concatenate([source_matrices[i] @ numpy.diag(weights_list[i]) for i in range(len(source_matrices))], axis=1)


def get_source_weighted_concat_coef(lmdas, myus, k, alpha):
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
    return numpy.sqrt(numpy.dot(lmdas[:k] ** (2 * alpha), myus[:k] ** (2 * alpha)) / numpy.dot(myus[:k] ** (2 * alpha), myus[:k] ** (2 * alpha)))


def get_dimension_weighted_concat_coef(lmdas, myus, k, alpha):
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
    #return (lmdas[:k] ** (2 * alpha)) / (myus[:k] ** (2 * alpha))  # old version
    return (lmdas[:k] ** alpha) / (myus[:k] ** alpha)

def save(fname, M):
    with open(fname, "wb") as F:
        numpy.save(F, M)
    pass
    

def check_vocabs(A, B):
    """
    Check whether vocabularies are identical.
    """
    assert(len(A) == len(B))
    for word in A:
        print(A[word])
        assert(A[word] == B[word])
    for word in B:
        assert(B[word] == A[word])
    
    pass


def process():
    #corpus_fname = "./data/text8.zip"
    corpus_fname = sys.argv[1]
    settings = [("glove","./config/glove_sample_config.yml"), ("word2vec", "./config/word2vec_sample_config.yml"), ("lsa", "./config/lsa_sample_config.yml")]
    mode = "all"

    cfg = {}
    path = {}
    signal_matrix = {}
    pip_calculator = {}
    tokenizer = {}
    alpha = 1

    for (algorithm, model_config) in settings:
        print(algorithm, model_config, corpus_fname)
        cfg[algorithm], path[algorithm], signal_matrix[algorithm], tokenizer[algorithm] = create_signal_matrix(corpus_fname, model_config, algorithm)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])     
    
    #check_vocabs(tokenizer["glove"].dictionary, tokenizer["word2vec"].dictionary)
    #check_vocabs(tokenizer["glove"].dictionary, tokenizer["lsa"].dictionary)
    #check_vocabs(tokenizer["word2vec"].dictionary, tokenizer["lsa"].dictionary)
    #check_vocabs(tokenizer["glove"].reversed_dictionary, tokenizer["word2vec"].reversed_dictionary)
    #check_vocabs(tokenizer["glove"].reversed_dictionary, tokenizer["lsa"].reversed_dictionary)
    #check_vocabs(tokenizer["word2vec"].reversed_dictionary, tokenizer["lsa"].reversed_dictionary)

    #save the vocabulary
    with open("vocab", "w") as F:
        for i in range(len(tokenizer["glove"].dictionary)):
            print("{0} = {1}".format(tokenizer["glove"].reversed_dictionary[i], i))
            F.write("%s\n" % tokenizer["glove"].reversed_dictionary[i])
        
    k = {}
    sources = []
    df = pd.DataFrame()
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        #k[algo] = 300
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ numpy.diag(signal_matrix[algo].spectrum[:k[algo]])
        sources.append(source_mat)    
        save("{0}.npz".format(algo), source_mat) 
        WR = WordReps()
        WR.load_matrix(source_mat, tokenizer[algo].dictionary)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode=mode), index=[algo]))

    
    print("Unweighted concatenation...")
    weights_list = [numpy.ones(k[algo]) for algo, _ in settings]
    M1 = concat(sources, weights_list)
    WR1 = WordReps()
    WR1.load_matrix(M1, tokenizer[algo].dictionary)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR1, mode=mode), index=["un"]))

    for alpha in numpy.linspace(0,5,21):
        print("alpha = {0}".format(alpha))
        df = batch_alpha(alpha, df, settings, sources, signal_matrix, pip_calculator, tokenizer, mode, k)    
   
    # save and display results
    df.to_csv("corpus-res.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))

def pairwise_evaluation():
    #corpus_fname = "./data/text8.zip"
    corpus_fname = sys.argv[1]
    settings = [("glove","./config/glove_sample_config.yml"), 
                ("word2vec", "./config/word2vec_sample_config.yml"), 
                ("lsa", "./config/lsa_sample_config.yml")]
    mode = "all"

    cfg = {}
    path = {}
    signal_matrix = {}
    pip_calculator = {}
    tokenizer = {}
    alpha = 1

    for (algorithm, model_config) in settings:
        print(algorithm, model_config, corpus_fname)
        cfg[algorithm], path[algorithm], signal_matrix[algorithm], tokenizer[algorithm] = create_signal_matrix(corpus_fname, model_config, algorithm)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])     

    #save the vocabulary
    with open("vocab", "w") as F:
        for i in range(len(tokenizer["glove"].dictionary)):
            print("{0} = {1}".format(tokenizer["glove"].reversed_dictionary[i], i))
            F.write("%s\n" % tokenizer["glove"].reversed_dictionary[i])
        
    k = {}
    sources = []
    df = pd.DataFrame()
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        #k[algo] = 300
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ numpy.diag(signal_matrix[algo].spectrum[:k[algo]])
        sources.append(source_mat)    
        save("{0}.npz".format(algo), source_mat) 
        WR = WordReps()
        WR.load_matrix(source_mat, tokenizer[algo].dictionary)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode=mode), index=[algo]))

    pairs = [(0,1), (0,2), (1,2)]    
    for (i,j) in pairs:
        print("Unweighted concatenation...")
        cur_settings = [settings[i], settings[j]]
        prefix = settings[i][0] + "+" + settings[j][0]
        weights_list = [numpy.ones(k[algo]) for algo, _ in cur_settings]
        cur_sources = [sources[i], sources[j]]
        M1 = concat(cur_sources, weights_list)
        WR1 = WordReps()
        WR1.load_matrix(M1, tokenizer["glove"].dictionary)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR1, mode=mode), index=["%s un" % prefix]))
        df = batch_alpha(2.0, df, cur_settings, cur_sources, signal_matrix, pip_calculator, tokenizer, mode, k, prefix)
  
   
    # save and display results
    df.to_csv("corpus-res.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def batch_alpha(alpha, df, settings, sources, signal_matrix, pip_calculator, tokenizer, mode, k, prefix=None):
    # source-weighted concatenation
    print("Source weighted concatenation...")
    weights_list = []
    for algo, _ in settings:
        c = get_source_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo], alpha)
        weights_list.append(c * numpy.ones(k[algo]))
    M2 = concat(sources, weights_list)
    WR2 = WordReps()
    WR2.load_matrix(M2, tokenizer[algo].dictionary)
    ind_str = "sw ({0:0.2f})".format(alpha)
    if prefix is not None:
        ind_str = prefix + " " + ind_str
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR2, mode=mode), index=[ind_str]))

    print("Dimension weighted concatenation...")
    weights_list = []
    for algo, _ in settings:
        c = get_dimension_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo], alpha)
        weights_list.append(c)
    M3 = concat(sources, weights_list)
    WR3 = WordReps()
    WR3.load_matrix(M3, tokenizer[algo].dictionary)
    ind_str = "dw ({0:0.2f})".format(alpha)
    if prefix is not None:
        ind_str = prefix + " " + ind_str
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR3, mode=mode), index=[ind_str]))
    return df


def main():
    M = numpy.random.randn(10, 50)
    words = []
    eval.evaluate_embed_matrix(M, words)
    pass

if __name__ == "__main__":
    process()
    #pairwise_evaluation()
    #main()