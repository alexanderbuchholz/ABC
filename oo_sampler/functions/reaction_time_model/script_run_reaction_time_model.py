# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:44:09 2016
    script that runs the reaction time model
@author: alex
"""

import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/reaction_time_model")

import numpy as np
import random
from functions_smc import *
from functions_reaction_time import *
from functions_file import *

if __name__ == "__main__":
    # define parameters
    # specify the relevant inputs
    barrier1 = 0.5
    drift1 = 0.001
    barrier2 = 0.7
    drift2 = 0.002
    h_discrete = 0.01
    T_max = 10
    s = 1
    array_input_true = np.array([barrier1, barrier2, s, drift1, drift2, drift2, drift1])
    simulate_reaction_time = processes_race_class(barrier1, barrier1, drift1, drift1, h_discrete, T_max, s)
    y_star = simulate_reaction_time.simulate_extract(array_input_true)
    y = simulate_reaction_time.simulate_extract(array_input_true)
    print delta_reaction_time(y,y_star)

    N_particles = 300
    epsilon = np.array([0.1, 0.05, 0.03, 0.02, 0.01, 0.005])
    smc_abc_rqmc = smc_sampler_abc(epsilon, N_particles, delta_reaction_time, y_star, simulate_reaction_time.simulate_extract, random_sequence_mc, gaussian_kernel, exclude_theta, covar_factor = 0.25, dim_theta=7, IS=True, multiple_y = 1)
    smc_abc_rqmc.initialize_sampler(theta_prior_rqmc)
    #import cProfile
    #cProfile.run('smc_abc_rqmc.loop_over_time(move_theta)')
    smc_abc_rqmc.loop_over_time(move_theta)
    print smc_abc_rqmc.ESS
    print np.average(smc_abc_rqmc.thetas[:,:,4], axis=1, weights=smc_abc_rqmc.weights[0,:,4])
    print array_input_true


import matplotlib.pyplot as plt
# the histogram of the data
n, bins, patches = plt.hist(smc_abc_rqmc.thetas[0,:,4], 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
plt.close()