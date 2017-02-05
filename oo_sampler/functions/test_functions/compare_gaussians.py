import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")

import numpy as np
from functions_smc import gaussian
from functions_file import random_sequence_mc
from numba import jit

def brownian_self(steps):
    u = random_sequence_mc(steps)
    brownian_increments = np.cumsum(gaussian(u))
    return brownian_increments

def brownian_numpy(steps):
    gaussian = np.random.normal(size=steps)
    brownian_increments = np.cumsum(gaussian)
    return brownian_increments

if __name__ == '__main__':
    steps = 100000
    brownian_self(steps)
    brownian_numpy(steps)
    import cProfile
    cProfile.run('brownian_self(steps)')
    cProfile.run('brownian_numpy(steps)')