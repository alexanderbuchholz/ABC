# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:43:57 2016
Create functions to sample from reaction time
@author: alex
"""
from __future__ import division
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
import random
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
randtoolbox = rpackages.importr('randtoolbox')

import numpy as np
from functions_smc import gaussian
from functions_file import random_sequence_mc
import matplotlib.pyplot as plt
import pdb
from numba import jit, autojit

#@profile
def brownian_path(T_max, h_discrete, m_drift):
    '''
    function that simulates a path with drift
    '''
    steps = np.ceil(T_max/h_discrete)
    #u = random_sequence_mc(steps)
    #gaussian_v = gaussian(u)
    # rather use the numpy version, it is 5 times faster than our own implementation
    gaussian_v = np.random.normal(size=steps)
    brownian_increments = np.cumsum(gaussian_v)
    path = m_drift*np.linspace(0,T_max,steps) + np.sqrt(h_discrete)*brownian_increments
    return path

def stopping_time(barrier, path):
    time = np.argmax(barrier < path)
    return time+(time==0)*path.size


class processes_race_class():
    """
        create a class for the racing processes
    """
    def __init__(self, barrier1, barrier2, drift1, drift2, h_discrete, T_max, s, N_simulations=1000):
        self.barrier1 = barrier1
        self.barrier2 = barrier2
        self.drift1 = drift1
        self.drift2 = drift2
        self.h_discrete = h_discrete
        self.T_max = T_max
        self.s = s
        self.N_simulations = N_simulations

    def simulate_path(self):
        self.path1 = brownian_path(self.T_max, self.h_discrete, self.drift1)
        self.path2 = brownian_path(self.T_max, self.h_discrete, self.drift2)

    def processes_race(self):
        """
            Race of the two processes
            Add method if laps occurs
        """
        # laps event
        if np.random.uniform()<0.05:
            stop_base = np.random.uniform()*800
            self.stop_times = np.array([stop_base, stop_base])
            self.winner = np.random.choice(2)
        # normal simulation
        else:
            tau = np.random.normal(scale=np.exp(self.s))
            self.simulate_path()
            self.stop_times = np.array([stopping_time(self.barrier1+tau, self.path1), stopping_time(self.barrier2+tau, self.path2)])
            # specify procedure when no one wins: pick the highest accumulator
            #pdb.set_trace()
            if self.stop_times[0] == self.stop_times[1]:
                self.winner = np.argmax((self.path1[-1], self.path2[-1]))
            else:
                self.winner = np.argmin(self.stop_times)

    def corrupt_result(self, a=100, b=200):
        #tau = gaussian(u=random_sequence_mc(1), mu = 0, sigma = np.exp(np.array([self.c])))
        #tau = np.random.normal(scale=np.exp(self.c))
        add_time = a+random_sequence_mc(1)*(b-a)
        self.add_time = add_time.flatten()
        self.corrupted_times = self.stop_times +self.add_time

    def simulator_single(self):
        """
        one single simulation of a path
        """
        # redefine variables

        # call the function to simulate
        self.processes_race()
        self.corrupt_result()
        return np.array([self.winner, self.corrupted_times[self.winner]])

    def show_winner(self):
        print self.winner, self.corrupted_times[self.winner]

    def simulator(self, array_input_from_prior):
        """
            loops the simulator to get several realisations of one experiment
        """
        # specify input parameters
        self.barrier1 = array_input_from_prior[0]
        self.barrier2 = array_input_from_prior[1]
        self.s = array_input_from_prior[2]
        self.drift1 = array_input_from_prior[3]
        self.drift2 = array_input_from_prior[4]

        self.output = np.zeros((self.N_simulations, 2))
        for i_simulate in xrange(self.N_simulations):
            self.output[i_simulate,:] = self.simulator_single()

    def simulate_extract(self, array_input_from_prior):
        """
            runs the simulator and extracts the summary statistic
        """
        # run simulator
        self.simulator(array_input_from_prior)
        # create the summary statistic
        percentiles = [0.25, 0.5, 0.75]
        self.summary = np.hstack([np.mean(self.output[:,0]), np.percentile(self.output[self.output[:,0]==0,1], percentiles)/self.output[self.output[:,0]==0,1].mean(), np.percentile(self.output[self.output[:,0]==1,1], percentiles)/self.output[self.output[:,0]==1,1].mean()])
        #pdb.set_trace()
        return self.summary

    def simulate_extract_several_experiments(self, array_input_from_prior_multi):
        """
            function that implements several experiments
        """
        number_experiments = int((array_input_from_prior_multi.shape[0]-3)/2)
        #pdb.set_trace()
        if (number_experiments % 1) != 0:
            raise ValueError("wrong number of input parameters !")
        self.summary_multi = np.array([])
        for i_experiment in xrange(number_experiments):
            #pdb.set_trace()
            self.simulate_extract(np.hstack((array_input_from_prior_multi[:3], array_input_from_prior_multi[(3+i_experiment*2):(5+i_experiment*2)])))
            self.summary_multi = np.vstack([self.summary_multi, self.summary]) if self.summary_multi.size else self.summary
        return self.summary_multi

def theta_prior_mc(i, dim=5, n=1):
    random_seed = random.randrange(10**9)
    np.random.seed(seed=random_seed)
    u_generator = np.asarray(np.random.uniform(size=dim*n).reshape((n,dim)))
    #if u_generator.shape[1] != 5:
    #    raise ValueError("Input random generator has wrong size ! ")
    lam = gaussian(u_generator[:,0])
    delt = gaussian(u_generator[:,1])
    barrier1 = np.exp(lam)
    barrier2 = np.exp(lam+delt)
    s = gaussian(u_generator[:,2])
    drift1 = -0.1+u_generator[:,3]*0.2
    drift2 = -0.1+u_generator[:,4]*0.2
    if dim>5:
        extra_drift = -0.1+u_generator[:,5:]*0.2
        return np.expand_dims(np.hstack((np.array([barrier1, barrier2, s, drift1, drift2]), extra_drift.squeeze())),1)
    else:
        return np.expand_dims(np.array([barrier1, barrier2, s, drift1, drift2]),1)

def theta_prior_rqmc(i, dim=5, n=1):
    random_seed = random.randrange(10**9)
    u_generator = np.array(randtoolbox.sobol(n=n, dim=dim, init=(i==0), scrambling=1, seed=random_seed)) # randtoolbox for sobol sequence
    #if u_generator.shape[1] != 5:
    #    raise ValueError("Input random generator has wrong size ! ")
    lam = gaussian(u_generator[:,0])
    delt = gaussian(u_generator[:,1])
    barrier1 = np.exp(lam)
    barrier2 = np.exp(lam+delt)
    s = gaussian(u_generator[:,2])
    drift1 = -0.1+u_generator[:,3]*0.2
    drift2 = -0.1+u_generator[:,4]*0.2
    if dim>5:
        extra_drift = -0.1+u_generator[:,5:]*0.2
        return np.expand_dims(np.hstack((np.array([barrier1, barrier2, s, drift1, drift2]), extra_drift.squeeze())),1)
    else:
        return np.expand_dims(np.array([barrier1, barrier2, s, drift1, drift2]),1)

def delta_reaction_time(y, y_star):
    return np.linalg.norm((y-y_star), ord=1)

def exclude_theta(*args):
    return(1)

if __name__ == "__main__":
    # define parameters
    barrier1 = 0.05
    drift1 = 0.001
    barrier2 = 0.07
    drift2 = 0.002
    h_discrete = 0.01
    T_max = 10
    s = 1
    array_input = np.array([barrier1, barrier2, s, drift1, drift2])
    # test prior
    prior_test = theta_prior_mc(0)
    # initialization
    simulate_reaction_time = processes_race_class(barrier1, barrier1, drift1, drift1, h_discrete, T_max, s)
    y_star = simulate_reaction_time.simulate_extract(prior_test)
    y = simulate_reaction_time.simulate_extract(prior_test)
    prior_test = np.array([barrier1, barrier2, s, drift1, drift2, drift1, drift2])
    y = simulate_reaction_time.simulate_extract_several_experiments(prior_test)
    print delta_reaction_time(y,y_star)
    #simulate_reaction_time.show_winner()
    #print simulate_reaction_time.stop_times
    if True:
        plt.plot(simulate_reaction_time.path1)
        plt.axhline(y=barrier1, c="red")
        plt.plot(simulate_reaction_time.path2, ls="dashed")
        plt.axhline(y=barrier2, ls="dashed")
        plt.show()
