from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp

from six.moves import xrange  # pylint: disable=redefined-builtin


class SignalMatrix():
    def __init__(self):
        self._param_dir = "params/{}".format(self.__class__.__name__)
        sp.check_output("mkdir -p {}".format(self._param_dir), shell=True)
    
    @property
    def param_dir(self):
        return self._param_dir

    def estimate_signal(self, enable_plot=False):
        const = self.construct_matrix(self.M)
        U, D, V = np.linalg.svd(const)
        if enable_plot:
            plt.plot(D)
            plt.savefig('{}/sv.pdf'.format(self._param_dir))
            plt.close()
        self.spectrum = D
        self.U = U
        self.V = V
        with open("{}/sv.pkl".format(self._param_dir), "wb") as f:
            pickle.dump(self.spectrum, f)

    def estimate_noise(self):
        matrix_1 = self.construct_matrix(self.A)
        matrix_2 = self.construct_matrix(self.B)
        diff = matrix_1 - matrix_2
        self.noise = np.std(diff) * 0.5

    def export_estimates(self):
        with open("{}/estimates.yml".format(self._param_dir), "w") as f:
            f.write("lambda: {}\n".format("sv.pkl"))
            f.write("sigma: {}\n".format(self.noise))
            f.write("alpha: {}\n".format(0.5)) #symmetric factorization


    def construct_matrix(self):
        raise NotImplementedError


