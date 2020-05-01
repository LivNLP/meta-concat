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
import eval

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

def process():
    corpus_fname = "./data/small.txt"
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
    source = {}
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        source[algo] = signal_matrix[algo].U[:,:k[algo]] @ numpy.diag(signal_matrix[algo].spectrum[:k[algo]])
        print(source[algo].shape)        
        WR = WordReps()
        WR.load_matrix(source[algo], tokenizer[algo].dictionary)
        evaluate_embed_matrix(WR, mode="lex")

def main():
    M = numpy.random.randn(10, 50)
    words = []
    eval.evaluate_embed_matrix(M, words)
    pass

if __name__ == "__main__":
    process()
    #main()