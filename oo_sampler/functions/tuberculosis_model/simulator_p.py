# -*- coding: utf-8 -*-
import numpy.random as nr
import numpy as np
from scipy.stats import itemfreq, gamma, norm
import random
#import cProfile
#from numba import jit
#from numba import autojit
#from numba import float64, int32
import pdb

#@jit()
def multinomial_sample(weights):
        """
                
        """
        weights = np.add.accumulate(weights)
        u = nr.random()
        position = np.searchsorted(weights, u)
	#N = weights.shape[0]
        #position_vec = np.zeros(N)
        #position_vec[position] = 1
        return(position)

#@jit(int32(float64[:]))
def multinomial_sample_cum(cum_weights):
        """
                
        """
        u = nr.random()
        position = np.searchsorted(cum_weights, u)
        return(position)
def multinomial_sample_cum2(cum_weights):
        """
                
        """
        u = nr.random()
        position = np.searchsorted(cum_weights, u)
	#N = weights.shape[0]
        #position_vec = np.zeros(N)
        #position_vec[position] = 1
        return(position)

#@jit(float64[:](float64[:], int32))
def update_cum_sum_birth(X_cum, position):
	X_cum[position:] +=1.
	return X_cum

#@jit(nopython=True)
def update_cum_sum_death(X_cum, position):
	X_cum[position:] -= 1.
	return X_cum 
#@jit(nopython=True)
def update_cum_sum_mutation(X_cum, position_old, position_new):
	X_cum[position_old:position_new] -= 1.
	return X_cum
#@autojit
def calc_W_cum(X_cum, N_alive):
	return X_cum/N_alive


#@autojit()
def inner_while(theta_cum, G, X, death_counter, icounter, N_alive, X_cum, W_cum):
	    # sample position
            selector_geneotype = multinomial_sample_cum(W_cum)
            selector_event = multinomial_sample_cum(theta_cum)

            # birth procedure
            if selector_event==0:
                    X[selector_geneotype] += 1.0
		    N_alive += 1.
		    X_cum[selector_geneotype:] +=1.
		    #X_cum = update_cum_sum_birth(X_cum, selector_geneotype)
		    W_cum = X_cum/N_alive

            # death procedure
            if selector_event==1:
                    death_counter = death_counter+1
                    # check whether we do not kill the single individual
                    if (N_alive-1.)>0:
			X[selector_geneotype] -= 1.0
		        N_alive -= 1.
			X_cum[selector_geneotype:] -= 1.
			#X_cum = update_cum_sum_death(X_cum, selector_geneotype)
		        W_cum = X_cum /N_alive

                        if X[selector_geneotype]==0:
				G[selector_geneotype] -= 1.0
                    else:
                        icounter = icounter-1

	    # mutation procedure
            if selector_event==2:
                    # mutation : delete current value
                    select_index = np.argmin(X) # needs to be done before modifying the list !
                    #pdb.set_trace()
                    #print select_index
		    X[selector_geneotype] -= 1.0
		    if X[selector_geneotype]==0:
				G[selector_geneotype] -= 1.0

                    # mutation : addd new geneotype
                    G[select_index] += 1.0 # add geneotype
                    X[select_index] += 1.0
		    X_cum[selector_geneotype:select_index] -= 1.
		    #X_cum = update_cum_sum_mutation(X_cum, selector_geneotype, select_index)
		    W_cum = X_cum/N_alive

            icounter += 1
	    return G, X, death_counter, icounter, N_alive, X_cum, W_cum
#@jit()
def loop(theta, N):
        X = np.zeros(N) # number of geneotypes vector (pop size)
	X_cum = np.ones(N)
        G = np.zeros(N) # geneotypes vector
        X[0] = 1
        G[0] =  1
	N_alive = np.array([1.])
        #W = X/X.sum(axis=1) # population weight vector
        W_cum = np.copy(X_cum)
	theta_cum = np.cumsum(theta)/np.sum(theta)
        icounter = 1
        death_counter = 0
	for i in xrange(N):
		G, X, death_counter, icounter, N_alive, X_cum, W_cum = inner_while(theta_cum, G, X, death_counter, icounter, N_alive, X_cum, W_cum)
        #while icounter < N:
	#	 G, X, death_counter, icounter, N_alive, X_cum, W_cum = inner_while(theta_cum, G, X, death_counter, icounter, N_alive, X_cum, W_cum)
	return X

#@jit()
#@profile
#@jit(nopython=True)
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
        X = loop(theta, N)
        X_reduced = np.int_(X[X>0])
        l = []
        for i in range(0, np.size(X_reduced)):
            l.append(i)
        identificator = np.array(l)
        #identificator = np.array([ i for i in range(0, np.size(X_reduced)) ]) # continue here
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
#@jit()
def repeat_simulation(f, theta, N):
	for i in xrange(N):
		f(theta)
if __name__ == "__main__":
        x = [[8],[2],[2]]
        theta = np.array(x, dtype=float)
        test_result = simulator(theta)
	#cProfile.run('loop(theta,10000)')
        #cProfile.run('simulator(theta)')
        print(test_result)
	N = 10000
	X = np.ones(N) # number of geneotypes vector (pop size)
        G = np.zeros(N) # geneotypes vector
        X[0] = 1
        G[0] =  1
	N_alive = 1
        icounter = 1
        death_counter = 0
	X_cum = np.ones(N)
	N_alive = np.array([1.])
        W_cum = np.copy(X_cum)
	theta_cum = np.cumsum(theta)/np.sum(theta)
	#inner_while(theta_cum,G, X, death_counter, icounter, N_alive, X_cum, W_cum)
	#cProfile.run('repeat_simulation(simulator, theta, 10)')
	#cProfile.run('inner_while(theta_cum,G, X, death_counter, icounter, N_alive, X_cum, W_cum)')
	#multinomial_sample(np.ones(2)/2)
	#inner_while(theta_cum,G, X, death_counter, icounter, N_alive, X_cum, W_cum)
        import yappi
	yappi.start()
	loop(theta,10000)
	yappi.get_func_stats().print_all()
	#cProfile.run('simulator(theta)')
        yappi.start()
	repeat_simulation(simulator, theta, 10)
        yappi.get_func_stats().print_all()
