from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from matrix.signal_matrix import SignalMatrix

class Word2VecMatrix(SignalMatrix):

    def set_params(self, vocab, reverse_vocab, M, A, B):
        self.vocab = vocab 
        self.reverse_vocab = reverse_vocab
        self.M = M 
        self.A = A 
        self.B = B 
        pass


    def construct_matrix(self, Nij):
        vocab_size = len(self.vocab)
        k = 1 # negative samples per positive sample.
        Ni = np.sum(Nij, axis=1)
        Nj = np.sum(Nij, axis=0)
        tot = np.sum(Nij)
        with warnings.catch_warnings():
            """log(0) is going to throw warnings, but we will deal with it."""
            warnings.filterwarnings("ignore")
            Pij = Nij / tot 
            Pi = Ni / np.sum(Ni)
            Pj = Nj / np.sum(Nj)
            # c.f.Neural Word Embedding as Implicit Matrix Factorization, Levy & Goldberg, 2014
            PMI = np.log(Pij) - np.log(np.outer(Pi, Pj)) - np.log(k)
            PMI[np.isinf(PMI)] = 0
            PMI[np.isnan(PMI)] = 0
        return PMI

