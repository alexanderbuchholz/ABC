import numpy as np
import ipdb as pdb

import os
root_path = "/home/alex/python_programming/ABC_results_storage/simulation_results_18-7-17"
os.chdir(root_path)
import sys
import copy
import pickle


from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid", {'axes.grid' : False})

#sys.path.append(root_path+"/ABC/oo_sampler/functions")
#sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
#sys.path.append(root_path+"/ABC/oo_sampler/functions/lotka_volterra_model")


m_list = [2,5,10,20, 40, 80]
i_list = range(49)
var_during = []
boxplot_mean_list_during = []
boxplot_mean_list_after = []
var_after = []
selector = 1
for m in m_list:

    mean_list_after = []
    mean_list_during = []
    for i in i_list:
        results = pickle.load(open('mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_m'+str(m)+'_'+str(i)+'_QMC'+str(m)+'_AIS_5000_simulation_abc_epsilon_1.p', 'rb'))
        #pdb.set_trace()
        mean_list_after.append(results['means_normalisation_constant'][:,-selector].mean())
        mean_list_during.append((1./results['N']*(results['means_normalisation_constant'][:,-selector]*(1-results['means_normalisation_constant'][:,-selector])).mean()))#*test_1['M_list'][0]
        #mean_list_during.append(results['variances_normalisation_constant'][:,-selector].mean()*results['M_list'][0])
    #pdb.set_trace()
    var_during.append(np.array(mean_list_during).mean())
    boxplot_mean_list_during.append(np.array(mean_list_during))
    boxplot_mean_list_after.append(np.array(mean_list_after))
    var_after.append((np.array(mean_list_after).var()*results['M_list'][0]))
    #print test_1['particles'][:,:,-1].mean(axis=1)
    #print test_1['particles'][:,:,-1].var(axis=1)
    
pdb.set_trace()
#plt.boxplot(boxplot_mean_list_during)
#plt.boxplot(boxplot_mean_list_after)
#plt.plot(var_after)
plt.plot(m_list, var_during, label="Var QMC", linewidth=3, linestyle='dashed')
plt.plot(m_list, var_after, label='Var standard', linewidth=3)
plt.legend(fontsize=14)
plt.yscale('log'); plt.xscale('log')
plt.ylabel('Variance times M', fontsize=14); plt.xlabel('M', fontsize=14)
plt.savefig('variance_during_after_several_m.png')
plt.show()
pdb.set_trace()



