"""
evaluation of the static simulation via violon plots
"""

import pickle
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import ipdb as pdb
import numpy as np

dim_max = 8
mc_results = np.zeros((dim_max, 2, 50))
qmc_results = np.zeros((dim_max, 2, 50))
rqmc_results = np.zeros((dim_max, 2, 50))
results_list = []
for dim in range(dim_max):
    simulation = pickle.load(open("static_simulation_gaussian_mixuture_dim"+str(dim+1)+".p", "rb"))
    mc_results = pd.DataFrame(simulation[0][:, 0, :, :].sum(axis=2).transpose(), columns=['mean', 'var'])
    mc_results['type'] = 'mc'
    qmc_results = pd.DataFrame(simulation[1][:, 0, :, :].sum(axis=2).transpose(), columns=['mean', 'var'])
    qmc_results['type'] = 'qmc'
    rqmc_results = pd.DataFrame(simulation[2][:, 0, :, :].sum(axis=2).transpose(), columns=['mean', 'var'])
    rqmc_results['type'] = 'rqmc'

    frames = [mc_results, qmc_results, rqmc_results]
    result = pd.concat(frames)
    result['dim'] = dim+1
    results_list.append(result)

results_total = pd.concat(results_list)
results_total['log_var'] = np.log(results_total['var'])

sns.violinplot(x="dim", y="mean", hue="type", data=results_total, palette="muted")
plt.savefig("violinplot_of_mean_estimator_several_dim"+".png")
plt.show()
sns.violinplot(x="dim", y="log_var", hue="type", data=results_total, palette="muted"); plt.show()

"""
simulations_list[0][0].shape
(2, 20, 50, 1)
first component : mean and var
second component : quantiles decreasing from 0.1 to 0.005
third component : iterations
fourth component : dimensions of the particle


simulation : list of three components
first element : mc
second element : qmc
third element : rqmc

length_quantiles = 20
quantiles = np.linspace(0.1, 0.005, num=length_quantiles)


set as target quantile the 0.01 quantile, corresponds to index 18
"""

pdb.set_trace()
