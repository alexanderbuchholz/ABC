# evaluation of tuberculosis simulation

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ipdb as pdb

sim_results_qmc = pickle.load(open('tuberculosis_model_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_QMC2_AIS_500_simulation_abc_epsilon_0.01.p', 'rb'))
sim_results_del_moral = pickle.load(open('tuberculosis_model_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_9_MC10_Del_Moral_500_simulation_abc_epsilon_0.01.p', 'rb'))

sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_palette("husl")
#pdb.set_trace()

x_del = sim_results_del_moral['particles'][0,:,-1] 
y_del = sim_results_del_moral['particles'][1,:,-1]

x_qmc = sim_results_qmc['particles'][0,:,-1] 
y_qmc = sim_results_qmc['particles'][1,:,-1]


"""with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")"""
#sns.jointplot(x=x_del, y=y_del, stat_func=None)
#sns.jointplot(x=x_qmc, y=y_qmc, stat_func=None)
pdb.set_trace()
plt.scatter(x=x_del, y=y_del, label="Del Moral", alpha = 0.5, marker='^', color="red")
plt.scatter(x=x_qmc, y=y_qmc, label='QMC', alpha= 0.5, marker='o', color="blue")
plt.legend(fontsize=14)
plt.xlabel('Birth rate', fontsize=14)
plt.ylabel('Death rate', fontsize=14)
plt.savefig('scatter_plot_tuberculosis.png')
plt.show()
