from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import warnings

from matrix.signal_matrix import SignalMatrix

class GloVeMatrix(SignalMatrix):

    def inject_params(self, kwargs):
        self._params = kwargs
        if "skip_window" not in self._params:
            self._params["skip_window"] = 5
        self.check_params()
        self.x_max = float(self._params["x_max"])
        self.alpha = float(self._params["alpha"])

    def check_params(self):
        if isinstance(self._params["skip_window"], int) and self._params["skip_window"] > 0:
            pass
        else:
            raise ValueError("skip_window must be a positive integer")

    def build_cooccurance_dict(self, data):
        skip_window = self._params["skip_window"]
        vocabulary_size = self.vocabulary_size
        cooccurance_count = collections.defaultdict(collections.Counter)
        for idx, center_word_id in enumerate(data):
            if center_word_id > vocabulary_size:
                vocabulary_size = center_word_id
            for i in range(max(idx - skip_window - 1, 0), min(idx + skip_window + 1, len(data))):
                cooccurance_count[center_word_id][data[i]] += 1
            cooccurance_count[center_word_id][center_word_id] -= 1
        return cooccurance_count, vocabulary_size

    def f(self, x):
        """
        Returns the square root of the f weight function for glove.
        """
        return np.log(x) * (1.0 if x > self.x_max else np.sqrt((x / self.x_max) ** self.alpha)) 


    def construct_matrix(self, data):
        cooccur, vocabulary_size = self.build_cooccurance_dict(data)

        Nij = np.ones([vocabulary_size, vocabulary_size])
        fvec = np.vectorize(self.f)
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                Nij[i,j] += cooccur[i][j]
        with warnings.catch_warnings():
            """log(0) is going to throw warnings, but we will deal with it."""
            # c.f. Pennington et al
            log_count = np.log(Nij) 
            #log_count = fvec(Nij)
        return log_count

