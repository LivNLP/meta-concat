from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from matrix.signal_matrix import SignalMatrix

class GloVeMatrix(SignalMatrix):

    def set_params(self, vocab, reverse_vocab, M, A, B):
        self.vocab = vocab 
        self.reverse_vocab = reverse_vocab
        self.M = M 
        self.A = A 
        self.B = B 
        pass

    def construct_matrix(self, cooc_matrix):
        with warnings.catch_warnings():
            """log(0) is going to throw warnings, but we will deal with it."""
            return np.log(cooc_matrix + 1)

