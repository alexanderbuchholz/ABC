# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:31:59 2016
     functions for the tuberculosis model
@author: alex
"""
from __future__ import division
#import cProfile
#import ipdb as pdb
import pickle
import random
import numpy.random as nr
import numpy as np

import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/lotka_volterra_model/R_simulator")

#from numba import jit

import rpy2.robjects.packages as rpackages
randtoolbox = rpackages.importr('randtoolbox')



model_string = "lotka_volterra_model"
dim = 3
exponent = 4    

path_archive_simulations = '/home/alex/python_programming/ABC_results_storage/models_information'


from simulator_r_based_lv import simulator
#import ipdb; ipdb.set_trace()
from simulator_r_based_lv import y_star_function



def transform_u_to_prior(u):
    """
    function that transform u to the correct prior
    """
    #import pdb; pdb.set_trace()
    u = np.exp(u*np.array([8.])+np.array([-6.]))
    return u


def theta_sampler_mc(i, dim, n, *args, **kwargs):
    """
    mc sampler for the prior generation of the tuberculosis example
    :param i: input counter, needed for rqmc initialisation
    :return: np.array of size 3, normalized theta
    """
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u = np.asarray(nr.uniform(size=dim*n).reshape((n, dim)))
    u = transform_u_to_prior(u)
    # sample gamma dist, theta1 is birth event proba
    return u.transpose() # Normalize theta in the end

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
    u = transform_u_to_prior(u)
    return u.transpose()

def theta_sampler_qmc(i, dim, n,*args, **kwargs):
    """
    rqmc sampler for the prior generation of the tuberculosis example
    :param i: input counter, needed for rqmc initialisation
    :return: np.array of size 3, normalized theta
    """
    random_seed = random.randrange(10**9)

    u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=0, seed=random_seed)) # randtoolbox for sobol sequence
    # sample gamma dist, theta1 is birth event proba
    #print u
    u = transform_u_to_prior(u)
    return u.transpose()


def delta(y_star, y):
    """
    Function to calculate the distance function of y and y star for the acceptance step
    :param y_star:  observed data
    :param y: simulated data
    :return: returns float difference according to distance function
    """
    dif_y = np.linalg.norm(y_star-y, ord=2)
    return(dif_y)

def exclude_theta(theta_prop):
    """
    function that excludes the theta values if not within the prior range
    """
    
    alpha1 = theta_prop[0]
    alpha2 = theta_prop[1]
    alpha3 = theta_prop[2]
    if np.array([alpha1 < np.exp(-6), alpha1 > np.exp(2)]).any():
        #import pdb; pdb.set_trace()
        return 0
    if np.array([alpha2 < np.exp(-6), alpha2 > np.exp(2)]).any():
        return 0
    if np.array([alpha3 < np.exp(-6), alpha3 > np.exp(2)]).any():
        return 0
    else: return 1


#N = 10 # number of particles
#initial_particles_mc = np.array([ theta_sampler_mc(0) for i in range(N)])
#y_star = np.array([0,0])
def f_y_star(dim=3):
    if dim != 3:
        raise ValueError('dimension needs to be 3 for lotka volterra')
    y_star = y_star_function()
    return y_star

def load_precomputed_data(dim, exponent):
    import os
    current_path = os.getcwd()
    os.chdir(path_archive_simulations)
    #import ipdb as pdb; pdb.set_trace()
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'rb') as handle:
        precomputed_data = pickle.load(handle)
    os.chdir(current_path)
    return precomputed_data

def precompute_save_data(exponent, dim):
    n = 10**exponent
    y_star = f_y_star()
    theta_array = theta_sampler_rqmc(i=0, dim=dim, n=n)
    y_diff_array = np.zeros(n)
    import os
    current_path = os.getcwd()
    os.chdir(path_archive_simulations)
    for i in xrange(n):
        #pdb.set_trace()
        y_diff_array[i] = delta(y_star, simulator(theta_array[:, i]))
    precomputed_data = {'theta_values': theta_array, 'y_diff_values': y_diff_array}
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'wb') as handle:
        pickle.dump(precomputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.chdir(current_path)

def epsilon_target(dim):
    if dim == 3:
        return 500
    else:
        raise ValueError('epsilon_target not available for the chosen dimension')
    

if __name__ == '__main__':
    #pdb.set_trace()
    theta_test = theta_sampler_rqmc(0, dim, 1)
    y = simulator(theta_test)
    y_star = f_y_star()
    #import ipdb; ipdb.set_trace()
    print delta(y,y_star)
    # save the true data
    plot_result = True
    if plot_result:
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid", {'axes.grid' : False})
        sns.set_palette("husl")
        import ipdb; ipdb.set_trace()
        
        test = load_precomputed_data(dim, exponent)
        selector = np.argmin(test['y_diff_values'])
        min_theta = test['theta_values'][:,selector]
        N = 1000
        y_values = np.zeros((N,16,2))
        for i in range(N):
            y_values[i,:,:] = simulator(min_theta)
        
        ipdb.set_trace()
        sns.tsplot(y_values[:,:,0]); plt.show()
        ipdb.set_trace()
        #true_theta = np.array([ 1.00626479,  0.68721715,  0.02604606])
        y = simulator(min_theta)
        plt.plot(y_star[:,0], linewidth=3, linestyle='dashed', label="y*")
        plt.plot(y_star[:,1], linewidth=3, linestyle='dashed')
        plt.plot(y[:,0], linewidth=3, label="y")
        plt.plot(y[:,1], linewidth=3)
        plt.legend(fontsize=14)
        plt.xlabel("t", fontsize=14)
        plt.ylabel("X", fontsize=14)
        plt.savefig('timeseries_lotka_volterra.png')
        plt.show()

        x = np.linspace(0, 15, 31)
        data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
        ax = sns.tsplot(data=data)





    if False:
        test = load_precomputed_data(dim, exponent)
        import pdb; pdb.set_trace()

    precompute_values = False
    if precompute_values:
        precompute_save_data(exponent, dim)
    save_y_star = True
    if save_y_star:
        y_star = f_y_star()
        import pickle
        with open(model_string+'_'+'y_star.p', 'wb') as handle:
            pickle.dump(y_star, handle, protocol=pickle.HIGHEST_PROTOCOL)