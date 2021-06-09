"""
This file implements the PIP-loss-based concatenated meta-embedding method.
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
PATH_TO_REPSEVAL = "../repseval/"
sys.path.insert(0, PATH_TO_REPSEVAL)
from evaluate import evaluate_embed_matrix
from wordreps import WordReps

def create_signal_matrix(model_config, algorithm, alpha):
    with open(model_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    corpus_fname = cfg.get('corpus')
    print("corpus file =", corpus_fname)
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
    signal_matrix.estimate_signal(enable_plot=False) # plot the singular values
    signal_matrix.estimate_noise()
    signal_matrix.export_estimates(alpha)
    return cfg, path, signal_matrix, tokenizer

def estimate_pip(path):
    pip_calculator = MonteCarloEstimator()
    pip_calculator.get_param_file(path, "estimates.yml")
    pip_calculator.estimate_signal()
    pip_calculator.estimate_pip_loss()
    #pip_calculator.plot_pip_loss()
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


def get_source_weighted_concat_coef(lmdas, myus, k, alpha, start_ind):
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
    
     start_ind : int
        start index of the lambda spectrum.
        
    Returns
    --------
    c : float
        the concatenation coefficient.
    """
    w = numpy.sqrt(numpy.dot(lmdas[start_ind : start_ind + k] ** (2 * alpha), myus[:k] ** (2 * alpha)) / numpy.dot(myus[:k] ** (2 * alpha), myus[:k] ** (2 * alpha))) 
    
    #upper = numpy.sum(lmdas[start_ind:start_ind+k] ** (2 * alpha))
    #lower = numpy.sum(myus[:k] ** (4 * alpha))
    #w = numpy.power(upper / lower, 1.0 / 3.0)
    #w = w / numpy.sum(w)

    return w


def get_dimension_weighted_concat_coef(lmdas, myus, k, alpha, start_ind):
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
    
    alpha : float
        alpha value for the source embedding.

    start_ind : int
        start index of the lambda spectrum.
        
    Returns
    --------
    c : numpy.array
        the concatenation coefficient.
    """
    return (lmdas[:k] ** alpha) / (myus[:k] ** alpha)  # old version (considers square root)
    #w = numpy.power(lmdas[start_ind : start_ind + k] ** (2 * alpha) / myus[:k] ** (4 * alpha), 1.0 / 3.0) # considers cubic root
    #w = w / numpy.sum(w)
    return w

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


def batch_process():
    """
    Calls process with different alphas.
    """
    corpus_fname = "./data/text8"
    for alpha in [0.75]:
        output_fname = "alpha={0}-res.csv".format(alpha)
        process(corpus_fname=corpus_fname, output_fname=output_fname, alpha=alpha)
    pass


def process(output_fname="", alpha=0):
    output_fname = sys.argv[1]
    alpha = float(sys.argv[2])

    print(output_fname, alpha)
    settings = [("glove","./config/glove_sample_config.yml"), ("word2vec", "./config/word2vec_sample_config.yml"),("lsa", "./config/lsa_sample_config.yml")]
    mode = "lex"

    cfg = {}
    path = {}
    signal_matrix = {}
    pip_calculator = {}
    tokenizer = {}
    #alpha = 0.5 # symmetric factorization

    
    vocab = set() # Union of all source vocabularies
    for (algorithm, model_config) in settings:
        print(algorithm, model_config)
        cfg[algorithm], path[algorithm], signal_matrix[algorithm], tokenizer[algorithm] = create_signal_matrix(model_config, algorithm, alpha)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])    
        vocab = vocab.union(set(tokenizer[algorithm].dictionary.keys())) 
    
    #check_vocabs(tokenizer["glove"].dictionary, tokenizer["word2vec"].dictionary)
    #check_vocabs(tokenizer["glove"].dictionary, tokenizer["lsa"].dictionary)
    #check_vocabs(tokenizer["word2vec"].dictionary, tokenizer["lsa"].dictionary)
    #check_vocabs(tokenizer["glove"].reversed_dictionary, tokenizer["word2vec"].reversed_dictionary)
    #check_vocabs(tokenizer["glove"].reversed_dictionary, tokenizer["lsa"].reversed_dictionary)
    #check_vocabs(tokenizer["word2vec"].reversed_dictionary, tokenizer["lsa"].reversed_dictionary)

    #save the vocabulary
    #with open("vocab", "w") as F:
    #    for i in range(len(tokenizer["glove"].dictionary)):
    #        print("{0} = {1}".format(tokenizer["glove"].reversed_dictionary[i], i))
    #       F.write("%s\n" % tokenizer["glove"].reversed_dictionary[i])
        
    k = {}
    unique_sources = []
    df = pd.DataFrame()
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ (numpy.diag(signal_matrix[algo].spectrum[:k[algo]] ** alpha))
        unique_sources.append(source_mat)

    # Different corpora will result in different vocabularies. We will compute the union of the vocabularies
    # and represent all signal matrices in the same vocabulary. If a word is not in a particular source we will
    # assign a zero vector for that word's corresponding source embedding.   
    sources = []
    print("Total unique words in the union of sources =", len(vocab))
    word_index = dict([(i,x) for (x,i) in enumerate(list(vocab))])
    print(word_index)
    
    for (algo, _), X in zip(settings, unique_sources):
        source_mat = numpy.zeros((len(word_index), k[algo])) 
        for word in word_index:
            if word in tokenizer[algo].dictionary:
                source_mat[word_index[word],:] = X[tokenizer[algo].dictionary[word],:]
        sources.append(source_mat)
        save("{0}.npz".format(algo), source_mat) 
        WR = WordReps()
        WR.load_matrix(source_mat, word_index)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode=mode), index=[algo]))
        WR.save_model("%s.embed" % algo)
    
    print("Unweighted concatenation...")
    weights_list = []
    for algo, _ in settings:
        weights_list.append(numpy.ones(k[algo]))
    M1 = concat(sources, weights_list)
    WR1 = WordReps()
    WR1.load_matrix(M1, word_index)
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR1, mode=mode), index=["un"]))
    WR1.save_model("uw.embeds")

    #for alpha in numpy.linspace(0,5,21):
    #    print("alpha = {0}".format(alpha))
    #    df = batch_alpha(alpha, df, settings, sources, signal_matrix, pip_calculator, tokenizer, mode, k)  
    df = batch_alpha(alpha, df, settings, sources, signal_matrix, pip_calculator, word_index, mode, k)    
   
    # save and display results
    df.to_csv(output_fname)
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
    alpha = 0.5

    for (algorithm, model_config) in settings:
        print(algorithm, model_config, corpus_fname)
        cfg[algorithm], path[algorithm], signal_matrix[algorithm], tokenizer[algorithm] = create_signal_matrix(corpus_fname, model_config, algorithm, alpha)
        pip_calculator[algorithm] = estimate_pip(path[algorithm])     

    k = {}
    sources = []
    df = pd.DataFrame()
    for algo, _ in settings:
        print(algo)
        k[algo] = numpy.argmin(pip_calculator[algo].estimated_pip_loss)
        #k[algo] = 300
        source_mat = signal_matrix[algo].U[:,:k[algo]] @ numpy.diag(signal_matrix[algo].spectrum[:k[algo]] ** alpha)
        sources.append(source_mat)    
        #save("{0}.npz".format(algo), source_mat) 
        WR = WordReps()
        WR.load_matrix(source_mat, tokenizer[algo].dictionary)
        df = df.append(pd.DataFrame(evaluate_embed_matrix(WR, mode=mode), index=[algo]))
    
    # uncomment the following loop if you do not want to l2 normalise the sources
    for j in range(len(sources)):
        sources[j] = numpy.divide(sources[j].T, numpy.linalg.norm(sources[j], axis=1)).T 

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
        df = batch_alpha(alpha, df, cur_settings, cur_sources, signal_matrix, pip_calculator, tokenizer, mode, k, prefix)
  
   
    # save and display results
    df.to_csv("corpus-res.csv")
    print(tabulate(df, headers='keys', tablefmt='psql'))


def batch_alpha(alpha, df, settings, sources, signal_matrix, pip_calculator, word_index, mode, k, prefix=None):
    # source-weighted concatenation
    print("Source weighted concatenation...")
    weights_list = []
    start_ind = 0

    for algo, _ in settings:
        c = get_source_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo], alpha, start_ind)
        weights_list.append(c * numpy.ones(k[algo]))
        start_ind += k[algo]

    M2 = concat(sources, weights_list)
    WR2 = WordReps()
    WR2.load_matrix(M2, word_index)
    ind_str = "sw ({0:0.2f})".format(alpha)
    if prefix is not None:
        ind_str = prefix + " " + ind_str
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR2, mode=mode), index=[ind_str]))
    WR2.save_model("sw.embeds")

    print("Dimension weighted concatenation...")
    weights_list = []
    start_ind = 0
    for algo, _ in settings:
        c = get_dimension_weighted_concat_coef(signal_matrix[algo].spectrum, 
                                            numpy.array(pip_calculator[algo].estimated_signal), 
                                           k[algo], alpha, start_ind)
        weights_list.append(c)
        start_ind += k[algo]
    M3 = concat(sources, weights_list)
    WR3 = WordReps()
    WR3.load_matrix(M3, word_index)
    ind_str = "dw ({0:0.2f})".format(alpha)
    if prefix is not None:
        ind_str = prefix + " " + ind_str
    df = df.append(pd.DataFrame(evaluate_embed_matrix(WR3, mode=mode), index=[ind_str]))
    WR3.save_model("dw.embeds")
    return df


def evaluate_pretrained(embed_fname):
    """
    Evaluate pretrained embeddings (sources/meta-embeddings).
    """
    #embed_fname = "glove.embed"
    res_fname = "./work/eval.csv"
    WR = WordReps()
    WR.read_model(embed_fname)
    df = pd.DataFrame()
    res = evaluate_embed_matrix(WR, mode="lex")
    df = df.append(pd.DataFrame(res, index=["embed"]))
    df.to_csv(res_fname)
    print(df)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    pass

def evaluate_multiple_embeddings():
    embeds = sys.argv[1:]
    for embed in embeds:
        print(embed)
        evaluate_pretrained(embed)
    pass


if __name__ == "__main__":
    process()
    #evaluate_multiple_embeddings()
    #batch_process()
    #pairwise_evaluation()