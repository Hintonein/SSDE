import numpy as np
import torch
import os

import time
from ssde import PDESymbolicSolver

def main():
    np.random.seed(100)
    MU = 1
    LEFT_BC, RIGHT_BC = 0, 1
    N_SAMPLES = 20 # collect n_samples points in the domain
    N_BOUNDARY = 20 # collect n_boundary points on the boundary
    X = np.random.uniform(LEFT_BC, RIGHT_BC, (N_SAMPLES, 1))
    X_bc = np.array([[LEFT_BC],[RIGHT_BC]])
    assert X_bc.shape == (2, 1)
    X_combine = np.concatenate([X, X_bc], axis=0)
    assert X_combine.shape == (N_SAMPLES+2, 1)
    y_bc = np.array([[0],[1]])
    y = np.zeros_like(X_combine)
    X_input = [X, X_bc]
    y_input = [y, y_bc] # bc points are computed twice in y and y_bc
    model = PDESymbolicSolver("./config/config_vanderpol.json")
    start_time = time.time()
    model.fit(X_input, y_input) # Should solve in ~10 hours
    print(model.program_.pretty())
    print(model.program_.sympy_expr)
    print('Using time(s):', time.time()-start_time)

if __name__ == '__main__':
    torch.set_num_threads(1)  
    torch.set_num_interop_threads(1) 
    torch.multiprocessing.set_sharing_strategy('file_system')

    main()

