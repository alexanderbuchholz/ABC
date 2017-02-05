# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:03:32 2016

test of different methods to sample from mixture distributions via qmc
@author: alex
"""

import numpy as np


from functions_transform_hilbert import *
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")

from functions_file import random_sequence_rqmc, random_sequence_mc
from functions_smc import gaussian


def multinomial_sampling(particles, weights, u):
    '''
        applies the sampling according to the multinomial sampling routine
    '''
    n, dim = particles.shape
    #array_index = array_transform_to_hilbert_index(squeeze_to_01(test))

    #order_u = np.argsort(u[:,0])
    sorted_u = np.sort(u[:,0])
    normalised_weigths = weights/np.sum(weights)
    resampled_indices = resampling_inverse_transform(sorted_u, normalised_weigths)
    #print resampled_indices
    test_t = np.zeros((n,dim))
    for j in range(n):
        test_t[j,:] = gaussian(u[j,1:],mu=particles[resampled_indices[j],:],sigma=np.array([0.1]))
    return test_t

def hilbert_sampling(particles, weights, u):
    '''
        applies the sampling according to the hilbert curve
    '''
    n, dim = particles.shape
    #array_index = array_transform_to_hilbert_index(squeeze_to_01(test))

    #order_u = np.argsort(u[:,0])
    sorted_u = np.sort(u[:,0])
    normalised_weigths = weights/np.sum(weights)
    normalised_permuted_weights = normalised_weigths[hilbert_sort(particles)]
    resampled_indices = resampling_inverse_transform(sorted_u, normalised_permuted_weights)
    #print resampled_indices
    test_t = np.zeros((n,dim))
    for j in range(n):
        test_t[j,:] = gaussian(u[j,1:],mu=particles[resampled_indices[j],:],sigma=np.array([0.1]))
    return test_t

def repeat_sampling(particles, weights, u_generator, sampler, Number_repetitions):
    '''
        repeats the sampling
    '''
    n, dim = particles.shape
    results = np.zeros((n, dim, Number_repetitions))
    for k_repetition in range(Number_repetitions):
        u = u_generator(size_mv = dim+1, i= k_repetition, n = n)
        results[:,:,k_repetition] = sampler(test, weights, u)
    return results

def log_variance(data, n_range):
    n_points = len(range(100,(n_range+100),100))
    out_var = np.zeros((n_points,2))
    k = 0
    for range_point in range(100,(n_range+100),100):
        out_var[k,:] = np.log((data[:,:,:range_point].mean(axis=0)).var(axis=1))
        #pdb.set_trace()
        k = k+1
    return out_var

if __name__ == '__main__':
    n1 = 50
    n2 = 50
    n = n1 + n2
    dim = 2
    i = 0
    #u = np.random.random(size=(n,dim+1))
    #u = random_sequence_rqmc(size_mv = dim+1, i= i, n = n)
    test1 = np.random.normal(size=(n1,dim))*0.2
    test2 = np.random.normal(size=(n2,dim))*0.2 + np.array([2,2])
    test = np.vstack((test1,test2))
    weights = np.ones(n)/n
    Number_repetitions = 100
    #test_t = hilbert_sampling(test, weights, u)
    if True:
        test_t_rqmc = repeat_sampling(test, weights, random_sequence_rqmc, hilbert_sampling, Number_repetitions)
        test_t_mc = repeat_sampling(test, weights, random_sequence_mc, hilbert_sampling, Number_repetitions)

        test_t_rqmc_mult = repeat_sampling(test, weights, random_sequence_rqmc, multinomial_sampling, Number_repetitions)
        test_t_mc_mult = repeat_sampling(test, weights, random_sequence_mc, multinomial_sampling, Number_repetitions)

        a1 = log_variance(test_t_rqmc, Number_repetitions)
        a2 = log_variance(test_t_mc, Number_repetitions)
        a3 = log_variance(test_t_rqmc_mult, Number_repetitions)
        a4 = log_variance(test_t_mc_mult, Number_repetitions)
        import matplotlib.pyplot as plt
        plt.plot(a1[:,1], color="blue")
        plt.plot(a2[:,1], color="red")
        plt.plot(a3[:,1], color="green")
        plt.plot(a4[:,1], color="black")
        plt.show()
    if False:
        import matplotlib.pyplot as plt
        test_t1 = repeat_sampling(test, weights, random_sequence_rqmc, hilbert_sampling, 1)
        test_t2 = repeat_sampling(test, weights, random_sequence_mc, multinomial_sampling, 1)
        plt.scatter(test[:,0], test[:,1], color='blue')
        plt.scatter(test_t1[:,0], test_t1[:,1], color='red')
        plt.scatter(test_t2[:,0], test_t2[:,1], color='green')
        plt.show()
