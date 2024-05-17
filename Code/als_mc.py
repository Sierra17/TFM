# Implementation of Alternating Minimization for Low Rank Matrix Completion LRMC
# based on SoftImpute ALS in section 3 of Hastie, Mazumder Lee and Zadeh (2015)
# Solving the biconvex problem ||A - UV^T||_F iteratively by least squares

import numpy as np
from scipy.linalg import orth, qr, lstsq
from scipy.sparse import coo_array


# Calculate error over mask, for different error options
def norm_diff (A, U, V, mask, option='l2'):
    diff_matrix = A - U @ V.transpose
    diff_values = diff_matrix[mask]
    match option:
        case 'l1':
            return np.sum(abs(diff_values))/np.count_nonzero(mask)
        case 'l2':
            return np.linalg.norm(diff_values)/np.count_nonzero(mask)
        case 'inf':
            return max(abs(diff_values))


# Initialize the orthogonal U matrix of dimensions n x k as a random matrix.
# A warm start is also possible by taking the QR decomposition of the domain matrix
def init_U (n, k):
    aux = np.random.random(size=(n, k))
    # U, _ = scipy.linalg.orth(aux)
    U, _ = qr(aux) # test which of the two options is faster
    return U


# Obtains V from U by least squares using V = (U^TU)^-1U^TA
def get_V_from_U (A, U):
    VT = lstsq(U, A, lapack_driver='gelsy') # But A has missing entries
    return np.transpose(VT)


def solve (A, n, m, k, mask, max_iter):
    U = init_U(n, k)
    for i in range(max_iter)
    return X


def main (A, n, m, k, p, max_iter):
    seed(1234)
    np.random.seed(1234)

    mask = np.random.choice(a=[True, False], size=(n, m), p=[p, 1-p])

    solve(A, n, m, k, Omega, max_iter)


if __name__ == '__main__':
    A = np.genfromtxt('M.txt', delimiter=',') #input matrix from Matlab as plaintext or csv
    n, m = np.shape(A)
    k = 5 
    p = 0.95
    main(A, )
