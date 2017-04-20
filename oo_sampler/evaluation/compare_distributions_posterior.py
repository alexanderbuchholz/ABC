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
path1 = "/home/alex/python_programming/ABC_results_storage/simulation_results_20-4-17_new"
path2 = "/home/alex/python_programming/ABC_results_storage/simulation_results"
import os
os.chdir(path1)
import sys
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/simulation")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/help_functions")
sys.path.append("/home/alex/python_programming/ABC/oo_sampler/functions/tuberculosis_model")
#import simulation_parameters_mixture_model_bimodal_gaussian_dim1_19_4_17_desktop as simulation_parameters_model
import simulation_parameters_mixture_model_gaussian_dim1_20_4_17_laptop as simulation_parameters_model
import f_rand_seq_gen
import gaussian_densities_etc
import resampling


from matplotlib import pyplot as plt
import seaborn as sns
def resample_particles(particles, weights):
    ancestors = resampling.residual_resample(weights)
    return(particles[ancestors])

if False:
    pdb.set_trace()
    simulation_RQMC = pickle.load( open("mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_38_RQMC2_AIS_2000_simulation_abc_epsilon_0.25.p", "rb"))
    simulation_QMC = pickle.load( open("mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_38_QMC2_AIS_2000_simulation_abc_epsilon_0.25.p", "rb"))
    simulation_MC = pickle.load( open("mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_38_MC2_AIS_2000_simulation_abc_epsilon_0.25.p", "rb"))
    simulation_MC_sisson = pickle.load( open("mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_38_MC1_true_sisson_2000_simulation_abc_epsilon_0.25.p", "rb"))
    simulation_MC_del_moral = pickle.load( open("mixture_gaussians_diff_variance_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_38_MC10_Del_Moral_2000_simulation_abc_epsilon_0.25.p", "rb"))

if False:
    pdb.set_trace()
    simulation_RQMC = pickle.load( open("mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_RQMC2_AIS_1000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_QMC = pickle.load( open("mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_QMC2_AIS_1000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_MC = pickle.load( open("mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_MC2_AIS_1000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_MC_sisson = pickle.load( open("mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_MC1_true_sisson_1000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_MC_del_moral = pickle.load( open("mixture_gaussians_bimodal_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_MC10_Del_Moral_1000_simulation_abc_epsilon_0.005.p", "rb"))

if False:
    rqmc_particles = resample_particles(simulation_RQMC['particles'][0,:,-1], simulation_RQMC['weights'][0,:,-1])
    qmc_particles = resample_particles(simulation_QMC['particles'][0,:,-1], simulation_QMC['weights'][0,:,-1])
    mc_particles = resample_particles(simulation_MC['particles'][0,:,-1], simulation_MC['weights'][0,:,-1])
    sisson_particles = resample_particles(simulation_MC_sisson['particles'][0,:,-1], simulation_MC_sisson['weights'][0,:,-1])
    del_moral_particles = resample_particles(simulation_MC_del_moral['particles'][0,:,-1], simulation_MC_del_moral['weights'][0,:,-1])

    plt.title("Posterior distribution of mixture of two gaussians")
    sns.kdeplot(rqmc_particles, label = 'rqmc')
    sns.kdeplot(qmc_particles,  label = 'qmc')
    sns.kdeplot(mc_particles, label = 'mc')
    sns.kdeplot(sisson_particles, label = 'sisson')
    g = sns.kdeplot(del_moral_particles, label = 'del moral')
    g.set(xlim=(-3, 3))
    plt.savefig('distribution_posterior_bimodel.png')
    plt.show()
    pdb.set_trace()

if True: 
    simulation_RQMC = pickle.load( open("single_gaussian_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_RQMC10_AIS_20000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_QMC = pickle.load( open("single_gaussian_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_QMC10_AIS_20000_simulation_abc_epsilon_0.005.p", "rb"))
    simulation_MC = pickle.load( open("single_gaussian_negative_binomial_uniform_kernel_1_VB_component_fixed_epsilon_schedule_algo_only_dim3_0_MC10_AIS_20000_simulation_abc_epsilon_0.005.p", "rb"))

    pdb.set_trace()
    
    simulation_RQMC['means_normalisation_constant'].mean(axis=0)
    simulation_QMC['means_normalisation_constant'].mean(axis=0)
    simulation_MC['means_normalisation_constant'].mean(axis=0)

    rqmc_vars_normalization_constant = simulation_RQMC['variances_normalisation_constant'].mean(axis=0)
    qmc_vars_normalization_constant = simulation_QMC['variances_normalisation_constant'].mean(axis=0)
    mc_vars_normalization_constant = simulation_MC['variances_normalisation_constant'].mean(axis=0)

    rqmc_budget = np.array(simulation_RQMC['M_list'])
    qmc_budget = np.array(simulation_QMC['M_list'])
    mc_budget = np.array(simulation_MC['M_list'])
    epsilons = simulation_MC['epsilon']

    plt.plot(simulation_RQMC['epsilon'], rqmc_budget*rqmc_vars_normalization_constant, label='rqmc')
    plt.plot(simulation_QMC['epsilon'], qmc_budget*qmc_vars_normalization_constant, label='qmc')
    plt.plot(simulation_MC['epsilon'], mc_budget*mc_vars_normalization_constant, label='mc')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=14)
    plt.xlabel('epsilon')
    plt.ylabel('Variance normalization constant times M')
    #plt.savefig('mse_variance_budget'+str(N_particles)+'N_variance_epsilon.png')
    plt.show()

    pdb.set_trace()