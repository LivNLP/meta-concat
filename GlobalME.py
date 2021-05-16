import numpy as np

def update(M_list, A, sources, eta):
    new_M_list = []
    for j in range(len(sources)):
        grad_M = (sources[j] - M_list[j] @ A) @ A.T 
        new_M_list.append(M_list[j] + 2 * eta * grad_M)
    return new_M_list

def compute_A(M_list, sources, d, n):
    T1 = np.linalg.inv(np.sum([M.T @ M for M in M_list], axis=0))
    T2 = np.zeros((d,n))
    for j in range(len(sources)):
        #print(M_list[j].shape, sources[j].shape)
        T2 += M_list[j].T @ sources[j]
    return T1 @ T2

def get_error(M_list, A, sources):
    return np.sum([np.linalg.norm(sources[j] - M_list[j] @ A) for j in range(len(sources))])

def GLME(sources, d, lr=0.01, epochs=100):
    """
    Implements the Globally Linear Meta-Embedding Learning method proposed by
    Yin+Shutze ACL 2016.

    For source embedding matrices E_1 and E_2, we learn their transformations M_1, M_2 and the
    meta-embedding A. The loss is give by
    \sum_{j=1}^{N} || E_j - M_j A||^2

    Update Equations are as follows:
    A = (M_1\TM_1 + M_2\TM_2)^{-1} (M_1\TE_1 + M_2\TE_2)

    grad_M1 = -2(E_1\T - M_1 A)A\T
    grad_M2 = -2(E_2\T - M_2 A)A\T

    Parameters
    -----------
    sources : list of numpy.ndarrays
        A list containing embedding matrices, which are numpy.ndarrays

    d : int
        dimensionality of the meta-embedding space


    Returns
    ----------
    M_list : list of numpy.ndarrays
        list of meta-embedding transformations, which are numpy.ndarrays
    
    A : numpy.ndarray
        Meta-embedding matrix
    """
    N = len(sources)    # number of sources
    n = sources[0].shape[1] # number of words in the vocabulary
    # all sources must have the same number of words
    for j in range(N):
        assert(sources[j].shape[1] == n)
    
    # Randomly initialise the transformation matrices
    M_list = []
    for j in range(N):
        M_list.append(np.random.randn(sources[j].shape[0], d))
    A = compute_A(M_list, sources, d, n)

    for i in range(epochs):
        M_list = update(M_list, A, sources, lr)
        A = compute_A(M_list, sources, d, n)
        error = get_error(M_list, A, sources)
        print("Epoch {0}: Error {1}".format(i, error))
    return M_list, A

    


    