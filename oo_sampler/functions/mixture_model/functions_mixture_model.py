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
import ipdb as pdb
#import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde


#from numba import jit
randtoolbox = rpackages.importr('randtoolbox')

model_string = "mixture_gaussians_diff_variance"
dim = 3
exponent = 6
#path_archive_simulations = '/home/alex/python_programming/ABC/oo_sampler/functions/mixture_model'
path_archive_simulations = '/home/alex/python_programming/ABC_results_storage/models_information'
var = 0.1

def simulator(theta):
    """
    Function that samples according to the birth death mutation process of Takana et al.
    :param theta: proposed parameter, obtained from the prior distribution simulation
    :param n_star: number of generated samples, in this function equal to one
    :return: freq_distinct_all : statistic for the observed genotype combination
    """
    # add random seed
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    unif_random = np.random.rand()
    if unif_random < 0.5:
        y = np.random.multivariate_normal(mean=np.atleast_1d(theta.squeeze()), cov = var*np.identity(theta.shape[0]))
    else:
        y = np.random.multivariate_normal(mean=np.atleast_1d(theta.squeeze()), cov = var*0.01*np.identity(theta.shape[0]))
    return y



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
        return u.transpose() # Normalize theta in the end

def theta_sampler_qmc(i, dim, n,*args, **kwargs):
        """
        qmc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)

        u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=0, seed=random_seed)).reshape((n,dim)) # randtoolbox for sobol sequence
        # sample gamma dist, theta1 is birth event proba
        #print u
        u = u*20-10
        return u.transpose()

def theta_sampler_rqmc(i, dim, n,*args, **kwargs):
        """
        rqmc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)

        u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=1, seed=random_seed)).reshape((n,dim)) # randtoolbox for sobol sequence
        # sample gamma dist, theta1 is birth event proba
        #print u
        u = u*20-10
        return u.transpose()

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
def f_y_star(dim=2):
    y_star = np.zeros(dim)
    return y_star


def true_posterior(theta):
    dim, N = theta.shape
    y_star = f_y_star(dim)
    density = 0.5*multivariate_normal.pdf(theta.transpose(), y_star, np.eye(dim)*var)+0.5*multivariate_normal.pdf(theta.transpose(), y_star, 0.01*np.eye(dim)*var)
    #pdb.set_trace()
    if density.shape[0] != N:
        raise ValueError('error in the dimensions of the input!')
    return density

def l1_distance(theta):
    #pdb.set_trace()
    selector = ~np.isnan(theta)
    theta = theta[:,selector[0,:]]
    estimated_kde = gaussian_kde(theta)
    evaluated_kde_points = estimated_kde.evaluate(theta)
    evaluated_posterior_points = true_posterior(theta)
    l1_distance_res = np.mean(abs(1-evaluated_posterior_points/evaluated_kde_points))
    #pdb.set_trace()
    return l1_distance_res


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
        #pdb.set_trace()
        y_diff_array[i] = delta(y_star, simulator(theta_array[:, i]))
    precomputed_data = {'theta_values': theta_array, 'y_diff_values': y_diff_array}
    import os
    os.chdir(path_archive_simulations)
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'wb') as handle:
        pickle.dump(precomputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def epsilon_target(dim):
    if dim == 1:
        return 0.005    # corresponds to the 0.05 percentile (0.0005) of 10**6 simulations 
                        # we keep 200 observations
    elif dim == 2:
        return 0.25
    elif dim == 3:
        return 1
    else:
        raise ValueError('epsilon target not available')
    


if __name__ == '__main__':
    precompute_values = False
    if precompute_values:
        precompute_save_data(exponent, dim)
    if False: 
        test = load_precomputed_data(1, 6)
        pdb.set_trace()


