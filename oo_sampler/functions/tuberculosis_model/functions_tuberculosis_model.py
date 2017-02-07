# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:31:59 2016
     functions for the tuberculosis model
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
#from numba import jit
from simulator_p import simulator
randtoolbox = rpackages.importr('randtoolbox')
StableEstim = rpackages.importr('StableEstim')

model_string = "tuberculosis_model"
import cython
#@jit
@cython.wraparound(False)
@cython.boundscheck(False)
def simulator2(theta):
        """
        Function that samples according to the birth death mutation process of Takana et al.
        :param theta: proposed parameter, obtained from the prior distribution simulation
        :param n_star: number of generated samples, in this function equal to one
        :return: freq_distinct_all : statistic for the observed genotype combination
        """
        # add random seed
        random_seed = random.randrange(10**9)
        np.random.seed(seed=random_seed)

        # Normalize theta
        if len(theta)<3:
            theta3 = 1-np.sum(theta)
            theta3 = theta3.reshape((1,1))
            theta = np.append(theta,theta3, axis = 0)
            theta = theta/np.sum(theta)
            # defining the intial probabilities for the events
        N = 10000 # number of simulated events
        N_subsample = 473 # number of subsamples that will be studied
            # t = 1
        X = np.zeros((1,N)) # number of geneotypes vector (pop size)
        G = np.zeros((1,N)) # geneotypes vector
        X[:,0] = 1
        G[:,0] = 1
        W = X/float(np.sum(X)) # population weight vector
        icounter = 1
        death_counter = 0
        while icounter < N:
            selector_geneotype = nr.multinomial(1,W.ravel(),1)==1
            selector_event = nr.multinomial(1,theta[:,0],1)==1
            # birth procedure
            if selector_event[:,0]==True:
                    X = X+selector_geneotype*1.0
            # death procedure
            if selector_event[:,1]==True:
                    death_counter = death_counter+1
                    # check whether we do not kill the single individual
                    if np.sum(X-selector_geneotype*1)>0:
                        X = X-selector_geneotype*1
                        if X[(selector_geneotype*1>0)&(selector_geneotype*1<0) ]==0:
                            #if X[np.nonzero(selector_geneotype*1)]==0:
                            G = G-selector_geneotype*1.0
                    else:
                        icounter = icounter-1
            if selector_event[:,2]==True:
                    # mutation : delete current value
                    select_index = np.argmin(X) # needs to be done before modifying the list !
                    X = X-selector_geneotype*1.0
                    if X[(selector_geneotype*1>0)&(selector_geneotype*1<0) ]==0:
                    #if X[np.nonzero(selector_geneotype*1)]==0:
                            G = G-selector_geneotype*1.0
                    # mutation : addd new geneotype
                    G[:,select_index]= 1.0 # add geneotype
                    X[:,select_index]= 1.0
            # Update the probability vector
            W = X/float(np.sum(X))
            icounter = icounter+1
        X_reduced = np.int_(X[X>0])
        identificator = np.array([ i for i in range(0, np.size(X_reduced)) ]) # continue here
        disaggregated_X = np.repeat(identificator, X_reduced, axis=0)
        # if statement in case that the death rate is too high
        if np.size(disaggregated_X)<N_subsample:
            #print "Attention, death rate too high, do not have enough samples\n"
            N_subsample = np.size(disaggregated_X)
        random_pop = np.random.choice(disaggregated_X, size=N_subsample, replace=False) # get random population
        freq_distinct_geneotypes = itemfreq(random_pop) # get the frequency of distinct geneotypes
        #first column: identifier of geneotype, second column: number of counts of this geneotype
        freq_distinct_all = itemfreq(freq_distinct_geneotypes[:,1]) # aggregate the information from before, sum over unique geneotypes
        #first column: population size geneotype, second column: number of geneotypes with the same population size
        # this corresponds to size m of clusters k
        return freq_distinct_all

if False:
	theta = np.array([[2],[1],[5]], dtype=float)
	#test_result = simulator(theta)
	cProfile.run('simulator(theta)')
	#print(test_result)

def first_fold(unif_sample):
    """
    Function that folds the first time
    :param unif_sample:
    :return: first folded samples
    """
    N = len(unif_sample)
    for i in range(N):
        if (unif_sample[i,0] < unif_sample[i,1]):
            inter_x = unif_sample[i,0]
            inter_y = unif_sample[i,1]
            unif_sample[i,0] = inter_y
            unif_sample[i,1] = inter_x
    return unif_sample

def second_fold(unif_sample):
    """
    function that folds the second time
    :param second_fold:
    :return:
    """
    N = len(unif_sample)
    for i in range(N):
        if (unif_sample[i,0] + unif_sample[i,1]) > 1:
            inter_x = unif_sample[i,0]
            inter_y = unif_sample[i,1]
            unif_sample[i,0] = 1-inter_y
            unif_sample[i,1] = 1-inter_x
    return unif_sample

def theta_sampler_mc(i=None, dim=2, n=1, *args, **kwargs):
        """
        mc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)
        np.random.seed(seed=random_seed)
        u = np.asarray(nr.uniform(size=dim*n).reshape((n,dim)))
        #print u
        # sample gamma dist, theta1 is birth event proba
        u = first_fold(second_fold(u))
        theta1 = u[:,0]
        # theta2 is death, always smaller than death probability
        theta2 = u[:,1]
        # theta3 is mutation
        #pmin = norm.cdf((-0.198/0.06735))
        #theta3 = (0.198)+(norm.ppf((u[2]*(1-pmin)+pmin))*0.06735) # simulation of truncated normal
        theta = np.array([theta1, theta2])#, (1-theta1-theta2)])
        return theta  # Normalize theta in the end

def theta_sampler_rqmc(i, dim=2, n=1, *args, **kwargs):
        """
        rqmc sampler for the prior generation of the tuberculosis example
        :param i: input counter, needed for rqmc initialisation
        :return: np.array of size 3, normalized theta
        """
        random_seed = random.randrange(10**9)
        u = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=1, seed=random_seed)) # randtoolbox for sobol sequence
        # sample gamma dist, theta1 is birth event proba
        #print u
        u = first_fold(second_fold(u))
        theta1 = u[:,0]
        # theta2 is death, always smaller than death probability
        theta2 = u[:,1]
        # theta3 is mutation
        #pmin = norm.cdf((-0.198/0.06735))
        #theta3 = (0.198)+(norm.ppf((u[2]*(1-pmin)+pmin))*0.06735) # simulation of truncated normal
        theta = np.array([theta1, theta2])#, (1-theta1-theta2)])
        return theta

def delta(y_star, y):
    """
    Function to calculate the distance function of y and y star for the acceptance step
    :param y_star:  observed data
    :param y: simulated data
    :return: returns float difference according to distance function
    """
    g_star = np.sum(y_star[:,1]) # number of different geneotypes in true sample
    n_i_star = y_star[:,0]

    g = np.sum(y[:,1])      # number of different geneotypes in simulated sample
    n_i = y[:,0]
    N_subsample = 473 # size of sampled population
    eta_star = 1 - np.sum((n_i_star/473)**2)
    eta = 1 - np.sum((n_i/N_subsample)**2)
    dif_y = abs(eta-eta_star)+abs(g-g_star)/N_subsample
    return(dif_y)

def exclude_theta(theta_prop):
    """
    """
    if len(theta_prop)>3:
        print theta_prop
        raise ValueError('theta is not consistent, size too large')
    if np.array([theta_prop[0]<theta_prop[1] , theta_prop[0]<0 , theta_prop[1]<0 , sum(theta_prop)>1]).any():
        return(0)
    else: return(1)

def check_consistency_theta(theta_prop):
    """
    function that checks the consistency of the proposed samples
    """
    if np.array([theta_prop[0]<theta_prop[1] , theta_prop[0]<0 , theta_prop[1]<0 , sum(theta_prop)>1]).any():
        raise ValueError('theta is not consistent, higher death than birth probability or no true probability')

# do we really need to complete theta prop ? only if passed to the sampler
def complete_theta(theta_prop):
    return(np.hstack((theta_prop, 1-sum(theta_prop))))
#N = 10 # number of particles
#initial_particles_mc = np.array([ theta_sampler_mc(0) for i in range(N)])

y_star = np.array([[1, 282],
                   [2, 20],
                   [3, 13],
                   [4, 4],
                   [5, 2],
                   [8, 1],
                   [10, 1],
                   [15, 1],
                   [23, 1],
                   [30, 1]], dtype=float)

def f_y_star(dim=None):
    if dim != 2:
        raise ValueError('dimension needs to be 2')
    return y_star

#y = simulator(initial_particles_mc[0,:,:])
#[simulator(initial_particles_mc[i,:,:]) for i in range(N)]

if __name__ == '__main__':
    #test = sample_function_test([1,0.1,1], n_star=None)
    #print test
    #print delta(1, test)
    import cProfile
    theta = np.array([[0.5],[0.1],[0.4]])
    cProfile.run('simulator(theta)')
    cProfile.run('simulator2(theta)')

    #unif_sample = first_fold(second_fold(unif_sample))
    #plt.plot(unif_sample[:,0],unif_sample[:,1], 'ro')
    #plt.show()
