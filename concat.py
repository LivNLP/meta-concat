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
    return numpy.dot(lmdas[:k], myus[:k]) / numpy.dot(myus[:k], myus[:k])


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
    #corpus_fname = "./data/text8.zip"
    corpus_fname = sys.argv[1]
    settings = [("glove","./config/glove_sample_config.yml"), ("word2vec", "./config/word2vec_sample_config.yml"), ("lsa", "./config/lsa_sample_config.yml")]

    cfg = {}
    path = {}
    signal_matrix = {}
    pip_calculator = {}
    tokenizer = {}

    for (algorithm, model_config) in settings:
        print(algorithm, model_config, corpus_fname)
        cfg[algorithm], path[algorithm], signal_matrix[algorithm], tokenizer[algorithm] = create_signal_matrix(corpus_fname, model_config, algorithm)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])     
    
    k = {}
    sources = []
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ numpy.diag(signal_matrix[algo].spectrum[:k[algo]])
        sources.append(source_mat)      
        WR = WordReps()
        WR.load_matrix(source_mat, tokenizer[algo].dictionary)
        evaluate_embed_matrix(WR, mode="lex")
    
    print("Unweighted concatenation...")
    weights_list = [numpy.ones(k[algo]) for algo, _ in settings]
    M = concat(sources, weights_list)
    WR = WordReps()
    WR.load_matrix(M, tokenizer[algo].dictionary)
    evaluate_embed_matrix(WR, mode="lex")

    # source-weighted concatenation
    print("Source weighted concatenation...")
    weights_list = []
    for algo, _ in settings:
        c = get_source_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo])
        weights_list.append(c * numpy.ones(k[algo]))
    M = concat(sources, weights_list)
    WR = WordReps()
    WR.load_matrix(M, tokenizer[algo].dictionary)
    evaluate_embed_matrix(WR, mode="lex")

    print("Dimension weighted concatenation...")
    weights_list = []
    for algo, _ in settings:
        c = get_dimension_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo])
        weights_list.append(c)
    M = concat(sources, weights_list)
    WR = WordReps()
    WR.load_matrix(M, tokenizer[algo].dictionary)
    evaluate_embed_matrix(WR, mode="lex")

def main():
    M = numpy.random.randn(10, 50)
    words = []
    eval.evaluate_embed_matrix(M, words)
    pass

if __name__ == "__main__":
    process()
    #main()