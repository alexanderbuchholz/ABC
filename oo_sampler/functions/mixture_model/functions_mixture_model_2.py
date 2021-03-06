# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:31:59 2016
     functions for the tuberculosis model
@author: alex
"""
import cProfile
import pickle
import numpy.random as nr
#import matplotlib
import numpy as np
import random
#import matplotlib.pyplot as plt
import pdb
#import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from scipy.stats import itemfreq, gamma, norm
#from numba import jit
randtoolbox = rpackages.importr('randtoolbox')
StableEstim = rpackages.importr('StableEstim')

model_string = "mixture_gaussians_bimodal"
dim = 1
exponent = 6
#path_archive_simulations = '/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model'
path_archive_simulations = '/home/alex/python_programming/ABC_results_storage/models_information'


def simulator(theta, fixed_seed=False):
        """
        Function that samples according to the birth death mutation process of Takana et al.
        :param theta: proposed parameter, obtained from the prior distribution simulation
        :param n_star: number of generated samples, in this function equal to one
        :return: freq_distinct_all : statistic for the observed genotype combination
        """
        # add random seed
        if fixed_seed == True:
            random_seed = 1
            np.random.seed(seed=random_seed)
        else:
            random_seed = random.randrange(10**9)
            np.random.seed(seed=random_seed)
        y1 = np.random.multivariate_normal(mean= -np.atleast_1d(theta.squeeze()), cov = np.identity(theta.shape[0]), size=50)
        y2 = np.random.multivariate_normal(mean=np.atleast_1d(theta.squeeze()), cov = np.identity(theta.shape[0]), size=50)
        y = np.vstack((y1,y2))
        q75, q25 = np.percentile(y, [75 ,25])
        iqr = q75 - q25
        return iqr



def theta_sampler_mc(i, dim, n,*args, **kwargs):
        """
        mc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)
        np.random.seed(seed=random_seed)
        u = np.asarray(nr.uniform(size=dim*n).reshape((n,dim)))
        u = u*20-10
        # sample gamma dist, theta1 is birth event proba
        return np.atleast_2d(u.transpose()) # Normalize theta in the end

def theta_sampler_rqmc(i, dim, n,*args, **kwargs):
        """
        rqmc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)

        u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=1, seed=random_seed)) # randtoolbox for sobol sequence
        # sample gamma dist, theta1 is birth event proba
        #print u
        u = u*20-10
        return np.atleast_2d(u.transpose())

def delta(y_star, y):
    """
    Function to calculate the distance function of y and y star for the acceptance step
    :param y_star:  observed data
    :param y: simulated data
    :return: returns float difference according to distance function
    """
    dif_y = np.linalg.norm(y_star-y)
    return(dif_y)

def exclude_theta(theta_prop):
    """
    """
    if np.array([theta_prop< -10 , theta_prop>10 ]).any():
        return(0)
    else: return(1)

def check_consistency_theta(theta_prop):
    """
    function that checks the consistency of the proposed samples
    """
    pass


#N = 10 # number of particles
#initial_particles_mc = np.array([ theta_sampler_mc(0) for i in range(N)])
#y_star = np.array([0,0])
def f_y_star(dim=1):
    #pdb.set_trace()
    y_star = simulator(np.array([1.]), fixed_seed=True)
    return y_star
#y_star = np.array([0,0])

    
def load_precomputed_data(dim, exponent):
    import os
    current_path = os.getcwd()
    os.chdir(path_archive_simulations)
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'rb') as handle:
        precomputed_data = pickle.load(handle)
    os.chdir(current_path)
    return precomputed_data

def precompute_save_data(exponent, dim):
    n = 10**exponent
    y_star = f_y_star(dim)
    theta_array = theta_sampler_rqmc(i=0, dim=dim, n=n)
    y_diff_array = np.zeros(n)
    for i in xrange(n):
        y_diff_array[i] = delta(y_star, simulator(theta_array[:, i]))
    precomputed_data = {'theta_values': theta_array, 'y_diff_values': y_diff_array}
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'wb') as handle:
        pickle.dump(precomputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    precompute_values = True
    if precompute_values:
        precompute_save_data(exponent, dim)
