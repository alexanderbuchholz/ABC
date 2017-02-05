# -*- coding: utf-8 -*-
# import necessary packages


from class_file import loop_simulation
from functions_file import simulator, theta_sampler_rqmc, delta, theta_sampler_mc
import numpy as np
import pickle

########################################
# specify input parameters #############
########################################

epsilon_list = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
#epsilon_list = [1,2]

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
    if True:
        N_list = [50,100,200,500,1000,2000]
        batches = 20
    if False:
        N_list = [5]
        batches = 2
    save_file = True
    type_sim = "mc"
    # construct the class
    tuberculosis_abc = loop_simulation(simulator, y_star, theta_sampler_mc, delta, epsilon, dim_theta, type_sim)
    # let it run !
    #tuberculosis_abc.loop_list_N(N_list,batches,save_file)
    #print tuberculosis_abc.mean_theta_list
    #print tuberculosis_abc.var_theta_list

from matplotlib import pyplot as plt



dimension_theta, dimension_y, nstar, theta_zero, K, B, N_range = 1, 1, 10, 2, 50, 1, [50,100,500,1000,2000,5000,10000]
epsilon = 1

out_rqmc = pickle.load( open( "rqmc_simulation_abc_epsilon_2.p", "rb" ) )
out_mc = pickle.load( open( "mc_simulation_abc_epsilon_2.p", "rb" ) )

var_rqmc = np.ravel(out_rqmc.theta_std_container)
var_mc = np.ravel(out_mc.theta_std_container)
#style.use('ggplot')
plt.plot([50,100,500,1000,2000,5000,10000], var_rqmc, label = "RQMC")
plt.plot([50,100,500,1000,2000,5000,10000], var_mc, label = "MC")
plt.ylabel('Standard deviation of estimated mean')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('logscale')
plt.ylabel('logscale')
plt.title('Standard deviation estimated mean qm and rqmc epsilon = 2')
plt.grid(True)
plt.legend()
plt.show()
