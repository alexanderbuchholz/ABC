# -*- coding: utf-8 -*-
# import necessary packages

import multiprocessing as mp
from class_file import loop_simulation
from functions_file import simulator, theta_sampler_rqmc, delta, theta_sampler_mc
import numpy as np


########################################
# specify input parameters #############
########################################

#epsilon_list = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
epsilon_list = [1,2]

def distributed_simulation(epsilon):
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
    dim_theta = 3
    if False:
        N_list = [50,100,200,500,1000]
        batches = 20
	save_file = True
    if True:
        N_list = [5]
        batches = 2
	save_file = False
    type_sim = "rqmc"
    # construct the class
    tuberculosis_abc = loop_simulation(simulator, y_star, theta_sampler_rqmc, delta, epsilon, dim_theta, type_sim)
    # let it run !
    tuberculosis_abc.loop_list_N(N_list,batches,save_file)
    print tuberculosis_abc.mean_theta_list
    print tuberculosis_abc.var_theta_list


pool = mp.Pool(processes=8)
pool.map(distributed_simulation, epsilon_list)
