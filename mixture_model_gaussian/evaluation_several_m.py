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

pdb.set_trace()

test = pickle.load(open('mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_m5_9_QMC5_AIS_1000_simulation_abc_epsilon_1.p', 'rb'))

