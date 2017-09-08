import ipdb as pdb
import cPickle as pickle
import numpy as np

import os
root_path = "/home/alex/python_programming"
import sys
import copy

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(root_path+"/ABC/oo_sampler/functions")

sys.path.append(root_path+"/ABC/oo_sampler/functions/help_functions")
sys.path.append(root_path+"/ABC/oo_sampler/functions/mixture_model")
sys.path.append(root_path+"/ABC")
from functions_static_simulation import *









if __name__ == '__main__':
    import functions_mixture_model as functions_model
    simulator = functions_model.simulator_vectorized
    delta = functions_model.delta_vectorized
    theta_sampler_mc = functions_model.theta_sampler_mc
    theta_sampler_rqmc = functions_model.theta_sampler_rqmc
    theta_sampler_qmc = functions_model.theta_sampler_qmc
    theta_sampler_list = [theta_sampler_mc, theta_sampler_qmc, theta_sampler_rqmc]
    sampler_type_list = ['MC', 'QMC', 'RQMC']

    dim_list = [1,2,4,8]
    dim_list = [1]
    m_intra = 15
    N_simulations = 10**6
    M_repetitions = 40
    #threshold_quantiles = np.linspace(10, 0.01, 100)
    threshold_quantiles = np.linspace(2, 0.01, 20)
    quantile_single = 0.1

    
    pdb.set_trace()
    vars_to_save = pickle.load(open('backup_means.p', "rb"))
    list_distributions_mc_var = vars_to_save[0]
    list_distributions_qmc_var = vars_to_save[1]
    list_distributions_rqmc_var = vars_to_save[2]
    plot_violin_plot(list_distributions_mc_var, list_distributions_qmc_var, list_distributions_rqmc_var, 'mean')
    #plot_violin_plot(list_distributions_mc_var, list_distributions_qmc_var, list_distributions_rqmc_var, 'var')
    pdb.set_trace()

    list_distributions_mc_mean = {}
    list_distributions_qmc_mean = {}
    list_distributions_rqmc_mean = {}
    list_distributions_mc_var = {}
    list_distributions_qmc_var = {}
    list_distributions_rqmc_var = {}
    for dim in dim_list:
        y_star = functions_model.f_y_star(dim)
        instance_compare_samplers = compare_sampling_methods(M_repetitions, simulator, delta, dim, N_simulations, y_star, m_intra=m_intra)

        instance_compare_samplers.simulate_and_extract(threshold_quantiles, quantile_single, target_function_mean, theta_sampler_list, sampler_type_list)
        name_plot = "mean_mixed_gaussian_static_dim_%s_m_%s.png" % (dim, m_intra)
        print 'now plotting'
        plot_variance_mean_variance(threshold_quantiles, instance_compare_samplers, name_plot)

        list_distributions_mc_mean[str(dim)] = instance_compare_samplers.distribution_results_mc
        list_distributions_qmc_mean[str(dim)] = instance_compare_samplers.distribution_results_qmc
        list_distributions_rqmc_mean[str(dim)] = instance_compare_samplers.distribution_results_rqmc

        instance_compare_samplers.simulate_and_extract(threshold_quantiles, quantile_single, target_function_var, theta_sampler_list, sampler_type_list)
        name_plot = "var_mixed_gaussian_static_dim_%s_m_%s.png" % (dim, m_intra)
        print 'now plotting'
        plot_variance_mean_variance(threshold_quantiles, instance_compare_samplers, name_plot)

        list_distributions_mc_var[str(dim)] = instance_compare_samplers.distribution_results_mc
        list_distributions_qmc_var[str(dim)] = instance_compare_samplers.distribution_results_qmc
        list_distributions_rqmc_var[str(dim)] = instance_compare_samplers.distribution_results_rqmc
    
    plot_violin_plot(list_distributions_mc_mean, list_distributions_qmc_mean, list_distributions_rqmc_mean, 'mean')
    plot_violin_plot(list_distributions_mc_var, list_distributions_qmc_var, list_distributions_rqmc_var, 'var')
    if False: 
        vars_to_save = [list_distributions_mc_var, list_distributions_qmc_var, list_distributions_rqmc_var]
        means_to_save = [list_distributions_mc_mean, list_distributions_qmc_mean, list_distributions_rqmc_mean]
        pickle.dump(vars_to_save, open('backup_vars.p', "wb"))
        pickle.dump(means_to_save, open('backup_means.p', "wb"))
    pdb.set_trace()


    test = False
    if test: 
        prior_values, distances = simulation_joint_distribution(
            simulator,
            delta,
            theta_sampler,
            dim,
            N_simulations,
            y_star,
            simulator_vectorized=True)

        reference_table_theta, reference_table_distances = repeat_joint_simulation(
            M_repetitions,
            simulator,
            delta,
            theta_sampler,
            dim,
            N_simulations,
            y_star,
            simulator_vectorized=True)

        variance_results = loop_extraction_reference_talbe_aggregated(
            reference_table_theta,
            reference_table_distances,
            threshold_list,
            target_function_mean)
