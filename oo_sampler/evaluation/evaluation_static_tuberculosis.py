# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:21:52 2017
    evaluation of results
@author: alex
"""
# pylint: disable=C0321
import ipdb as pdb
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
path1 = "/home/alex/inter_simulation_results/simulations_results_12_5_17_tuberculosis"
import os
os.chdir(path1)


with open('tuberculosis_modelmeans_static_simulation_gaussian_mixuture_dim2.p', 'rb') as handle:
    tuberculosis_results = pickle.load(handle)

simuluation_type = tuberculosis_results.keys()
mc_results = tuberculosis_results['mc']
qmc_results = tuberculosis_results['qmc']
rqmc_results = tuberculosis_results['rqmc']
M_repetitions = len(mc_results)
pdb.set_trace()
distances = mc_results[0]['distances'].flatten()
#quantiles_list = [0.5, 1, 2, 10]
quantiles_list = np.linspace(0.05, 10, 20)
thresholds  = np.percentile(distances, quantiles_list)
number_thresholds = len(quantiles_list)
means_result = np.zeros(shape=(2,M_repetitions,number_thresholds, 3))
#pdb.set_trace()

for j_treshold in range(number_thresholds):
    for i_repetition in range(M_repetitions):
        means_result[:, i_repetition, j_treshold, 0] = mc_results[i_repetition]['particles'][:, mc_results[i_repetition]['distances'].flatten()<thresholds[j_treshold]].mean(axis=1)
        means_result[:, i_repetition, j_treshold, 1] = qmc_results[i_repetition]['particles'][:, qmc_results[i_repetition]['distances'].flatten()<thresholds[j_treshold]].mean(axis=1)
        means_result[:, i_repetition, j_treshold, 2] = rqmc_results[i_repetition]['particles'][:, rqmc_results[i_repetition]['distances'].flatten()<thresholds[j_treshold]].mean(axis=1)

pdb.set_trace()

variances_posterior = means_result.var(axis=1)

# plot of the variance reduction
#plt.title('Variance reduction for the tuberculosis model', fontsize=18)
sns.set_palette("husl")
sns.set_style("whitegrid", {'axes.grid' : False})
plt.plot(quantiles_list, variances_posterior[1,:,0]/variances_posterior[1,:,1], linewidth=3, label='MC/QMC', linestyle='dashed')
plt.plot(quantiles_list, variances_posterior[1,:,0]/variances_posterior[1,:,2], linewidth=3, label='MC/RQMC', linestyle='dotted')
#plt.xscale('log')
plt.ylabel('Variance reduction factor', fontsize=14); plt.xlabel('Quantile of distance in percent', fontsize=14)
plt.legend(fontsize=14, loc=2)
plt.savefig('variance_reduction_tuberculosis.png')
plt.show()





import pandas as pd
#from pandas.plotting import scatter_matrix
df_mc = pd.DataFrame(mc_results[0]['particles'][:, mc_results[0]['distances'].flatten()<0.01].transpose(), columns=['1', '2'])
df_mc['type'] = 'mc'
df_qmc = pd.DataFrame(qmc_results[0]['particles'][:, qmc_results[0]['distances'].flatten()<0.01].transpose(), columns=['1', '2'])
df_qmc['type'] = 'qmc'
frames = [df_mc, df_qmc]
result = pd.concat(frames)
sns.pairplot(df_mc); plt.show()
sns.pairplot(df_qmc); plt.show()
sns.pairplot(result, hue='type', palette="husl", plot_kws={"s":40, "alpha":.5,'lw':1, 'edgecolor':'k'}); plt.show()



pdb.set_trace()
plt.boxplot([means_result[0,:,3,i] for i in range(3)]); plt.show()
plt.boxplot(means_result[0,:,0,1])

pdb.set_trace()