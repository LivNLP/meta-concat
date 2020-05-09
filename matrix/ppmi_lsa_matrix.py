from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from six.moves import xrange  # pylint: disable=redefined-builtin

from matrix.signal_matrix import SignalMatrix

class LSAMatrix(SignalMatrix):

    def set_params(self, vocab, reverse_vocab, M, A, B):
        self.vocab = vocab 
        self.reverse_vocab = reverse_vocab
        self.M = M 
        self.A = A 
        self.B = B 
        pass

    def construct_matrix_raw(self, cooccur):
        return cooccur

    def construct_matrix(self, Nij):
        vocab_size = len(self.vocab)
        Ni = np.sum(Nij, axis=1)
        tot = np.sum(Nij)
        with warnings.catch_warnings():
            """log(0) is going to throw warnings, but we will deal with it."""
            warnings.filterwarnings("ignore")
            Pij = Nij / tot 
            Pi = Ni / np.sum(Ni)
            # c.f.Neural Word Embedding as Implicit Matrix Factorization, Levy & Goldberg, 2014
            PMI = np.log(Pij) - np.log(np.outer(Pi, Pi))
            PMI[np.isinf(PMI)] = 0
            PMI[np.isnan(PMI)] = 0
            PPMI = PMI
            PPMI[PPMI < 0] = 0
        return PPMI

    

