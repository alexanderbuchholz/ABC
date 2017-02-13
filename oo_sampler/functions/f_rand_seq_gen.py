# -*- coding: utf-8 -*-
import cProfile
import numpy.random as nr
#import matplotlib
import numpy as np
#import matplotlib.pyplot as plt
import pdb
#import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from scipy.stats import itemfreq, gamma, norm
import random
#from numba import jit
#from simulator_p import simulator
randtoolbox = rpackages.importr('randtoolbox')
#StableEstim = rpackages.importr('StableEstim')



def random_sequence_qmc(size_mv, i, n=1):
    """
    generates QMC random sequence for the movement of particles
    still needs to be done
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=0, seed=random_seed)).reshape((n,size_mv))
    # randtoolbox for sobol sequence
    #pdb.set_trace()
    return(u)



def random_sequence_rqmc(size_mv, i, n=1):
    """
    generates QMC random sequence for the movement of particles
    still needs to be done
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    u = np.array(randtoolbox.sobol(n=n, dim=size_mv, init=(i==0), scrambling=1, seed=random_seed)).reshape((n,size_mv))
    # randtoolbox for sobol sequence
    #pdb.set_trace()
    return(u)

#@jit
#@profile
def random_sequence_mc(size_mv, i=None,n=1):
    """
    generates MC random sequence for the movement of particles
    """
    size_mv = np.int(size_mv)
    n = np.int(n)
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u = np.asarray(nr.uniform(size=size_mv*n).reshape((n,size_mv)))
    #pdb.set_trace()
    return(u)

# define the true data

