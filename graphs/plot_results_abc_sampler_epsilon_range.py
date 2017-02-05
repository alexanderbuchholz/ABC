# -*- coding: utf-8 -*-
# Plot results simulation ABC sampler
# first simulation mc
import abc_sampler
import pickle
import numpy as np
#from matplotlib import style
from matplotlib import pyplot as plt



dimension_theta, dimension_y, nstar, theta_zero, K, B, N_range = 1, 1, 10, 2, 50, 1, [1000,2000,5000,10000]
epsilon = [0.1,0.2,0.3,0.4,0.5,1,1.5,2]

counter_N_range = 0
for j_N in N_range:
    var_rqmc_list = np.zeros(len(epsilon)) # define empty lists
    var_mc_list = np.zeros(len(epsilon))
    counter_epsilon = 0
    # loop over epsilon for plot
    for i_epsilon in epsilon:
        out_rqmc = pickle.load( open( "new_rqmc_simulation_abc_epsilon_"+str(i_epsilon)+".p", "rb" ) )
        out_mc = pickle.load( open( "new_mc_simulation_abc_epsilon_"+str(i_epsilon)+".p", "rb" ) )
        var_rqmc_list[counter_epsilon] = np.ravel(out_rqmc.theta_std_container)[counter_N_range]
        var_mc_list[counter_epsilon] = np.ravel(out_mc.theta_std_container)[counter_N_range]
        counter_epsilon = counter_epsilon+1 # counter to loop over list
        
    counter_N_range = counter_N_range + 1 # add to counter
    plt.plot(epsilon, var_rqmc_list, label = "RQMC")
    plt.plot(epsilon, var_mc_list, label = "MC")
    plt.ylabel('Standard deviation of estimated mean')
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('logscale')
    plt.ylabel('logscale')
    plt.title('Standard deviation estimated mean qm \n and rqmc varying epsilon, N='+str(j_N))
    plt.grid(True)
    plt.legend()
    plt.savefig('std_estimated_mean_varying_eps_N'+str(j_N)+'.pdf')
    plt.clf()
