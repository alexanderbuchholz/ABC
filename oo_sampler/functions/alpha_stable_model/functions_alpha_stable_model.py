# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:22:42 2016
    functions for the alpha stable model
@author: alex
"""

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
from numba import jit
randtoolbox = rpackages.importr('randtoolbox')
StableEstim = rpackages.importr('StableEstim')

def simulator(theta, N=100):
    """
    function that samples form the alpha stable distribution
    theta is np.array([alpha, beta, gamma, delta])
    """
    # unpack values
    #	theta = theta.astype(object)
    alpha = theta[0,:]
    beta =  theta[1,:]
    gamma =  theta[2,:]
    delta =  theta[3,:]
    # add random seed
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    # generate w and u for simulating
    #	pdb.set_trace()
    w = nr.exponential(size=N)
    u = nr.uniform(low=-np.pi/2., high=np.pi/2., size=N)
    #	w = w.astype(float)
    #	u = u.astype(float)
    S_a_b = (1.+ beta**2. * np.tan(np.pi*alpha/2.)**2. )**(1/(2.*alpha))
    B_a_b = 1./alpha * np.arctan(beta*np.tan(np.pi*alpha*0.5))
    if alpha == 1.:
        y_bar = 2./np.pi * ((np.pi/2. + beta*u)*np.tan(u)-beta*np.log(np.pi/2. * w *np.cos(u ) / (np.pi/2. + beta*u)   ) )
    else:
        y_bar = S_a_b * ((np.sin(alpha)*(u + B_a_b ) ) / np.cos(u)**(1./alpha)  ) * (np.cos(u-alpha*(u+ B_a_b ))/w) **((1-alpha)/alpha )

    return S1_summary_statistic_alpha(y_bar*gamma+delta, theta)

def S1_summary_statistic_alpha(y, theta_true=None):
    """
    function that calculates the summary statistic, handles the case when theta is not available via an r package
    """
    if theta_true is None:
        # handle the case when there is no theta
        c = np.array(StableEstim.McCullochParametersEstim(x= robjects.vectors.FloatVector(y)))[2]
        v_alpha = (np.percentile(y, 95) - np.percentile(y,5))/ (np.percentile(y, 75)-np.percentile(y, 25))
        v_beta = (np.percentile(y, 95) + np.percentile(y,5)- 2*np.percentile(y,50))/ (np.percentile(y, 95)-np.percentile(y, 5))
        v_gamma =  (np.percentile(y, 75) - np.percentile(y,25))/c
        v_delta = np.mean(y)

    else:
        v_alpha = (np.percentile(y, 95) - np.percentile(y,5))/ (np.percentile(y, 75)-np.percentile(y, 25))
        v_beta = (np.percentile(y, 95) + np.percentile(y,5)- 2*np.percentile(y,50))/ (np.percentile(y, 95)-np.percentile(y, 5))
        v_gamma =  (np.percentile(y, 75) - np.percentile(y,25))/ theta_true[2,:]
        v_delta = np.mean(y)
    return np.array([v_alpha, v_beta, v_gamma, v_delta])

def theta_sampler_mc(i=None, dim=4, n=1, *args, **kwargs):
    """
    mc sampler for the prior generation of the alpha stable model
    :param i: input counter, needed for rqmc initialisation
    :return: np.array of size 4
    """
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u = np.asarray(nr.uniform(size=dim*n).reshape((n,dim)))
    #print u
    # sample gamma dist, theta1 is birth event proba
    alpha = u[:,0]*0.9+1.1
    beta = u[:,1]*2-1
    gamma = u[:,2]*300
    delta = u[:,3]*600-300
    theta = np.array([alpha, beta, gamma, delta])
    return theta

def theta_sampler_rqmc(i=None, dim=4, n=1, *args, **kwargs):
    """
    mc sampler for the prior generation of the alpha stable model
    :param i: input counter, needed for rqmc initialisation
    :return: np.array of size 4
    """
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=1, seed=random_seed)) # randtoolbox for sobol sequence
    #print u
    # sample gamma dist, theta1 is birth event proba
    alpha = u[:,0]*0.9+1.1
    beta = u[:,1]*2-1
    gamma = u[:,2]*300
    delta = u[:,3]*600-300
    theta = np.array([alpha, beta, gamma, delta])
    return theta

def delta(y_star, y):
    """
    Function to calculate the distance function of y and y star for the acceptance step
    :param y_star:  observed data
    :param y: simulated data
    :return: returns float difference according to distance function
    """
    dif_y = np.sum((y_star - y)**2)
    return(dif_y)

def exclude_theta(theta_prop):
    """
    exclude the theta values that do not fit in the space, unadmissable proposals
    """
    space = np.array([[1.1,2. ],[-1,1],[3,300],[-300,300]])
    if len(theta_prop)<3:
        print theta_prop
        raise ValueError('theta is not consistent, size too small')
    if not np.all((np.squeeze(theta_prop) < space[:,1]) & (np.squeeze(theta_prop) > space[:,0])):
        return(0)
    else: return(1)

y_star = np.array([5, -0.2, 1.4, -140])

if __name__ == '__main__':
    theta = theta_sampler_mc()
    y = simulator(theta)
    print y 
    print delta(y_star, y)
