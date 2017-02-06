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



#from numba import jit

import rpy2.robjects.packages as rpackages
randtoolbox = rpackages.importr('randtoolbox')

import rtnorm as rt


model_string = "toggle_switch_model"
dim = 7
exponent = 5

#path_archive_simulations = '/home/alex/python_programming/ABC/oo_sampler/functions/toggle_switch_model'
path_archive_simulations = '/home/alex/python_programming/ABC_results_storage/models_information'


from simulator_r_based import simulator


def simulator2(theta, T=300, N=500):
    """
    simulator for the toggle switch model
    """
    raise ValueError('this function is deprecated! ')
    alpha1 = theta[:, 0]
    alpha2 = theta[:, 1]
    beta1 = theta[:, 2]
    beta2 = theta[:, 3]
    mu = theta[:, 4]
    sigma = theta[:, 5]
    gamma = theta[:, 6]
    y = np.zeros(N)
    for n in range(N):
        u_t = 10
        v_t = 10
        for t in xrange(T):
            #pdb.set_trace()
            u_new = 0.97*u_t + alpha1/(1+v_t**beta1)-1
            u_lower = -u_new
            #u_new = u_new + 0.5*truncnorm.rvs(2*u_lower, np.inf)
            u_new = u_new + 0.5*rt.rtnorm(2*u_lower, np.inf)
            v_new = 0.97*v_t + alpha2/(1+u_t**beta2)-1
            v_lower = -v_new
            #v_new = v_new + 0.5*truncnorm.rvs(2*v_lower, np.inf)
            v_new = v_new + 0.5*rt.rtnorm(2*v_lower, np.inf)
            u_t = u_new
            v_t = v_new
        normal_error = nr.normal()
        y[n] = u_t +  mu + normal_error*mu*sigma/(u_t**gamma)
        #assert False
    # TODO: speedup truncated normal
    return y





def transform_u_to_prior(u):
    """
    function that transform u to the correct prior
    """
    #import pdb; pdb.set_trace()
    u = u*np.array([50., 50., 5., 5., 200., 0.5, 0.4])+np.array([0, 0, 0, 0, 250, 0, 0])
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

def delta(y_star, y):
    """
    Function to calculate the distance function of y and y star for the acceptance step
    :param y_star:  observed data
    :param y: simulated data
    :return: returns float difference according to distance function
    """
    dif_y = np.linalg.norm((np.sort(y_star)-np.sort(y))/y.shape[0], ord=2)
    return(dif_y)

def exclude_theta(theta_prop):
    """
    function that excludes the theta values if not within the prior range
    """
    alpha1 = theta_prop[0]
    alpha2 = theta_prop[1]
    beta1 = theta_prop[2]
    beta2 = theta_prop[3]
    mu = theta_prop[4]
    sigma = theta_prop[5]
    gamma = theta_prop[6]
    if np.array([alpha1 < 0., alpha1 > 50.]).any():
        return 0
    if np.array([alpha2 < 0., alpha2 > 50.]).any():
        return 0
    if np.array([beta1 < 0., beta1 > 5.]).any():
        return 0
    if np.array([beta2 < 0., beta2 > 5.]).any():
        return 0
    if np.array([mu < 250., mu > 450.]).any():
        return 0
    if np.array([sigma < 0., sigma > 0.5]).any():
        return 0
    if np.array([gamma < 0., gamma > 0.4]).any():
        return 0
    else: return 1


#N = 10 # number of particles
#initial_particles_mc = np.array([ theta_sampler_mc(0) for i in range(N)])
#y_star = np.array([0,0])
def f_y_star(dim=7):
    if dim != 7:
        raise ValueError('dimension needs to be 7 for toggle switch model')
    import os
    current_path = os.getcwd()
    os.chdir(path_archive_simulations)
    with open(model_string+'_'+'y_star.p', 'rb') as handle:
        y_star = pickle.load(handle)
    os.chdir(current_path)
    return y_star


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
    y_star = f_y_star()
    theta_array = theta_sampler_rqmc(i=0, dim=dim, n=n)
    y_diff_array = np.zeros(n)
    for i in xrange(n):
        #pdb.set_trace()
        y_diff_array[i] = delta(y_star, simulator(theta_array[:, i]))
    precomputed_data = {'theta_values': theta_array, 'y_diff_values': y_diff_array}
    with open(model_string+'_dim_'+str(dim)+'_npower_'+str(exponent)+'.p', 'wb') as handle:
        pickle.dump(precomputed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    #pdb.set_trace()
    theta_test = theta_sampler_rqmc(0, 7, 1)
    y = simulator(theta_test)
    y_star = f_y_star()
    print delta(y,y_star)
    # save the true data
    precompute_values = True
    if precompute_values:
        precompute_save_data(exponent, dim)
    if False:
        theta = np.array([22, 12, 4, 4.5, 325, 0.25, 0.15])
        y_star = simulator(theta)
        import pickle
        with open(model_string+'_'+'y_star.p', 'wb') as handle:
            pickle.dump(y_star, handle, protocol=pickle.HIGHEST_PROTOCOL)