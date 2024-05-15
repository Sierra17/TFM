# Implementation of Alternating Minimization for Low Rank Matrix Completion LRMC
# based on Hastie et al. (2015)

from os.path import dirname

from random import *
import math

import numpy as np
from scipy.linalg import orth


def make_random_lr_matrix (m, n, k):
    L = np.random.randn(m, k)
    R = np.random.randn(k, n)
    return np.dot(L, R)


# Mask given by omega as 0, 1
def get_masked_matrix (M, omega):
    M_max, M_min = np.max(M), np.min(M)
    M_ = M.copy()
    M_[(1 - omega).astype(np.double)] = M_max * M_max
    return M_


# Solving linear system as single iteration (convex)
def get_V_from_U (M, U, omega):
    column = M.shape[1]
    rank = U.shape[1]
    V = np.empty((rank, column), dtype=M.dtype)

    for j in range(0, column):
        U_ = U.copy()
        U_[(1 - omega[:, j]).astype(np.int16), :] = 0
        V[:, j] = np.linalg.lstsq(U_, M[:, j], rcond=None)[0] # Can think about conditioning for H matrices
    return V


# Matrix norm wrapper, finds miscalculated entries (zeros), fro = Frobenius, nuc = Nuclear
def get_error (M, U, V, omega):
    error_matrix = M - np.dot(U, V)
    error_matrix[(1 - omega).astype(np.int16)] = 0
    return np.linalg.norm(error_matrix, 'fro') / np.count_nonzero(omega)


def init_U (M, omega, p, k, mu):
    M[(1 - omega).astype(np.int16)] = 0
    M = M / p
    U, S, V = np.linalg.svd(M, full_matrices=False)

    U_hat = U.copy()
    clip_threshold = 2 * mu * math.sqrt(k / max(M.shape))
    U_hat[U_hat > clip_threshold] = 0
    U_hat = orth(U_hat)
    print("|U_hat-U|_F/|U|_F:",
          np.linalg.norm(np.subtract(U_hat, U), ord='fro') / np.linalg.norm(
              U, ord='fro'))
    return U_hat


def solve (M, omega, p, k, T, mu):
    U = init_U(M[:, :], omega, p, k, mu)
    print('')
    V = None
    for t in range(T):
        V = get_V_from_U(M, U, omega)
        U = get_V_from_U(M.T, V.T, omega.T).T
        err = get_error(M, U, V, omega)
        print('>> t(%3d):' % t, err)
    print('')
    assert V is not None
    return np.dot(U, V)


def main (m, n, k, p, T, mu, M):
    seed(1234)
    np.random.seed(1234)
    #M = make_random_lr_matrix(m, n, k)
    #my_dict = {}
    #scipy.io.loadmat('M', my_dict)
    #print(my_dict.keys)
    #M = my_dict['M']
    omega = np.zeros((m, n))
    omega[np.random.rand(m, n) <= p] = 1
    cardinality_of_omega = np.count_nonzero(omega)
    omega = omega.astype(np.int16) # using int16 could be improved
    M_rank = np.linalg.matrix_rank(M)
    print("RANK of M        :", M_rank)
    M_ = get_masked_matrix(M, omega)

    X = solve(M, omega, p, k, T, mu)
    X_rank = np.linalg.matrix_rank(X)
    print("RANK of X        :", X_rank)

    E = np.subtract(M, X)
    E_train = E.copy()
    np.place(E_train, 1 - omega, 0)
    print('RMSE       :',
          np.linalg.norm(E_train, "fro") / cardinality_of_omega)

    print("|X-M|_F/|M|_F    :",
          np.linalg.norm(np.subtract(M, X), ord='fro') / np.linalg.norm(
              M, ord='fro'))


if __name__ == '__main__':
    # m = 337
    # n = 337
    p = 0.95
    # Hyper Parameters: constrained rank, iterations, mu for thresholding
    k = 3
    T = 5
    mu = 0.1
    M = np.genfromtxt('input.csv', delimiter=',')#input matrix from Matlab
    n, m = np.shape(M)
    main(m, n, k, p, T, mu, M)
