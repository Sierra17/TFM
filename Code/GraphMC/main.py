import numpy as np
import mc_solver
import mc_run

gamma_n = 5
gamma_r = 0.003
gamma_c = 0.003

hparams = [gamma_n, gamma_r, gamma_c]

# data matrix to be completed, alternatively provide mask
X = np.array(input_matrix) 
if np.isnan(X).any():
    mask = np.isnan(X)

mc_run(X, mask, hparams)
