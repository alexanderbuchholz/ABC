# -*- coding: utf-8 -*-
# Plot results simulation ABC sampler
# first simulation mc
import abc_sampler
import pickle
import numpy as np
#from matplotlib import style
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
